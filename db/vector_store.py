"""
Vector store module for product similarity recommendations using Qdrant.

This module provides functionality to store and search product embeddings
using Qdrant vector database with BERT-based encodings.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import DistilBertModel, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)


class ProductVectorStore:
    """
    Vector store for product recommendations using Qdrant and DistilBERT embeddings.
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "products",
                 vector_size: int = 768):
        """
        Initialize the ProductVectorStore.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the collection to store product vectors
            vector_size: Size of the embedding vectors (768 for DistilBERT)
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            self.client = None
        
        # Initialize BERT model and tokenizer for encoding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.encoder.to(self.device)
            self.encoder.eval()
            logger.info(f"Loaded DistilBERT model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            self.tokenizer = None
            self.encoder = None
    
    def is_connected(self) -> bool:
        """Check if connected to Qdrant server."""
        if self.client is None:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the products collection in Qdrant.
        
        Args:
            recreate: Whether to recreate the collection if it exists
            
        Returns:
            True if collection was created/exists, False otherwise
        """
        if not self.client:
            logger.error("Qdrant client not available")
            return False
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def encode_text(self, text: str, max_length: int = 128) -> Optional[np.ndarray]:
        """
        Encode text using DistilBERT to get embedding vector.
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector as numpy array or None if encoding fails
        """
        if not self.tokenizer or not self.encoder:
            logger.error("BERT model not available for encoding")
            return None
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # FIX: Remove the extra 0.0 append that caused dimension mismatch
            return embedding.flatten()
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None
    
    def add_product(self, product_id: int, product_title: str, product_description: str = "") -> bool:
        """
        Add a product to the vector store.
        
        Args:
            product_id: Unique product ID
            product_title: Product title
            product_description: Product description (optional)
            
        Returns:
            True if product was added successfully, False otherwise
        """
        if not self.client:
            logger.error("Qdrant client not available")
            return False
        
        # Combine title and description for encoding
        text_to_encode = f"{product_title} {product_description}".strip()
        embedding = self.encode_text(text_to_encode)
        
        if embedding is None:
            logger.error(f"Failed to encode product: {product_title}")
            return False
        
        # FIX: Check if embedding has correct dimensions
        if embedding.shape[0] != self.vector_size:
            logger.error(f"Embedding dimension mismatch: expected {self.vector_size}, got {embedding.shape[0]}")
            return False
        
        try:
            point = PointStruct(
                id=product_id,
                vector=embedding.tolist(),
                payload={
                    "product_title": product_title,
                    "product_description": product_description,
                    "text": text_to_encode
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added product: {product_title} (ID: {product_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add product: {e}")
            return False
    
    def search_similar_products(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar products based on query text.
        
        Args:
            query: Search query text
            limit: Number of similar products to return
            
        Returns:
            List of similar products with scores
        """
        if not self.client:
            logger.error("Qdrant client not available")
            return []
        
        # Encode the query
        query_embedding = self.encode_text(query)
        if query_embedding is None:
            logger.error(f"Failed to encode query: {query}")
            return []
        
        try:
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "product_title": hit.payload.get("product_title", ""),
                    "product_description": hit.payload.get("product_description", ""),
                    "text": hit.payload.get("text", "")
                })
            
            logger.info(f"Found {len(results)} similar products for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            return []
    
    def add_sample_products(self):
        """Add some sample products for testing."""
        sample_products = [
            {
                "id": 1,
                "title": "MacBook Air M2",
                "description": "Fast and lightweight laptop with Apple M2 chip, 13-inch display, long battery life"
            },
            {
                "id": 2,
                "title": "Dell XPS 13",
                "description": "Premium ultrabook with Intel processors, compact design, high performance laptop"
            },
            {
                "id": 3,
                "title": "Lenovo Yoga Slim",
                "description": "Thin and light laptop with good performance, portable design, excellent for work"
            },
            {
                "id": 4,
                "title": "Gaming Desktop PC",
                "description": "High performance gaming computer with powerful graphics card and fast processor"
            },
            {
                "id": 5,
                "title": "Wireless Bluetooth Headphones",
                "description": "Noise canceling wireless headphones with long battery life and premium sound quality"
            },
            {
                "id": 6,
                "title": "4K Monitor",
                "description": "Large 27-inch 4K display monitor for professional work and entertainment"
            },
            {
                "id": 7,
                "title": "Mechanical Keyboard",
                "description": "RGB backlit mechanical gaming keyboard with tactile switches"
            },
            {
                "id": 8,
                "title": "Wireless Mouse",
                "description": "Ergonomic wireless mouse with precision tracking and long battery life"
            }
        ]
        
        # Create collection if it doesn't exist
        if not self.create_collection():
            logger.error("Failed to create collection for sample products")
            return
        
        # Add sample products
        successful_adds = 0
        for product in sample_products:
            if self.add_product(
                product_id=product["id"],
                product_title=product["title"],
                product_description=product["description"]
            ):
                successful_adds += 1
        
        logger.info(f"Added {successful_adds}/{len(sample_products)} sample products")
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        if not self.client:
            return None
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,  # FIX: Use collection name instead of size
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.name,  # FIX: Get distance name
                "points_count": info.points_count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None


# Global instance
vector_store = ProductVectorStore()