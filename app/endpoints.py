"""
API endpoints for the sentiment analysis service.

This module contains the route handlers for the FastAPI application.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Optional, List

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
    DetailedRecommendationResponse
)
from ml.model import ReviewClassifier
from db.vector_store import vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global classifier instance (will be set during app startup)
classifier: Optional[ReviewClassifier] = None


def get_classifier() -> ReviewClassifier:
    """
    Dependency to get the classifier instance.
    
    Returns:
        ReviewClassifier instance
        
    Raises:
        HTTPException: If classifier is not loaded
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server startup."
        )
    return classifier


def set_classifier(clf: ReviewClassifier):
    """
    Set the global classifier instance.
    
    Args:
        clf: ReviewClassifier instance
    """
    global classifier
    classifier = clf


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API service
    """
    model_ready = classifier is not None
    
    return HealthResponse(
        status="healthy" if model_ready else "unhealthy",
        model_ready=model_ready,
        message="Sentiment analysis API is running" if model_ready else "Model not loaded"
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(
    request: PredictionRequest,
    clf: ReviewClassifier = Depends(get_classifier)
):
    """
    Predict sentiment for a single text.
    
    Args:
        request: Request containing text to classify
        clf: ReviewClassifier instance (injected dependency)
        
    Returns:
        Prediction result with label and confidence
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Received prediction request for text: {request.text[:50]}...")
        
        # Make prediction using the classifier
        label, confidence = clf.predict(request.text)
        
        logger.info(f"Prediction completed: {label} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            label=label,
            confidence=confidence
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sentiment_batch(
    request: BatchPredictionRequest,
    clf: ReviewClassifier = Depends(get_classifier)
):
    """
    Predict sentiment for multiple texts.
    
    Args:
        request: Request containing list of texts to classify
        clf: ReviewClassifier instance (injected dependency)
        
    Returns:
        Batch prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Received batch prediction request for {len(request.texts)} texts")
        
        # Validate input
        if not request.texts:
            raise ValueError("Texts list cannot be empty")
        
        if len(request.texts) > 100:
            raise ValueError("Too many texts in batch. Maximum is 100.")
        
        # Make batch predictions
        results = clf.predict_batch(request.texts)
        
        # Convert to response format - FIXED: Ensure order is preserved
        predictions = []
        for i, (label, confidence) in enumerate(results):
            predictions.append(
                PredictionResponse(
                    label=label,
                    confidence=confidence
                )
            )
        
        logger.info(f"Batch prediction completed for {len(predictions)} texts")
        
        return BatchPredictionResponse(predictions=predictions)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction"
        )


@router.get("/model/info")
async def get_model_info(clf: ReviewClassifier = Depends(get_classifier)):
    """
    Get information about the loaded model.
    
    Args:
        clf: ReviewClassifier instance (injected dependency)
        
    Returns:
        Model information
    """
    try:
        model_info = clf.get_model_info()
        logger.info("Model info requested")
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model information"
        )


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_products(request: RecommendationRequest):
    """
    Get product recommendations based on search query.
    
    Args:
        request: Request containing search query text
        
    Returns:
        List of recommended product names
        
    Raises:
        HTTPException: If recommendation search fails
    """
    try:
        logger.info(f"Received recommendation request: {request.text}")
        
        # Check if vector store is available
        if not vector_store.is_connected():
            logger.warning("Qdrant vector store not available")
            raise HTTPException(
                status_code=503,
                detail="Product recommendation service is unavailable. Qdrant server may not be running."
            )
        
        # Ensure collection exists and has data
        try:
            collection_info = vector_store.get_collection_info()
            if collection_info is None or collection_info.get('points_count', 0) == 0:
                logger.info("Vector store collection is empty, adding sample products...")
                vector_store.add_sample_products()
        except Exception as e:
            logger.warning(f"Failed to check collection, creating new one: {e}")
            vector_store.create_collection()
            vector_store.add_sample_products()
        
        # Search for similar products
        similar_products = vector_store.search_similar_products(
            query=request.text,
            limit=3
        )
        
        if not similar_products:
            logger.info("No similar products found")
            return RecommendationResponse(recommended_products=[])
        
        # Extract product titles - FIXED: Handle potential missing keys
        product_names = []
        for product in similar_products:
            title = product.get("product_title", "")
            if title:
                product_names.append(title)
        
        logger.info(f"Found {len(product_names)} product recommendations")
        
        return RecommendationResponse(recommended_products=product_names)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during product recommendation"
        )


@router.post("/recommend/detailed", response_model=DetailedRecommendationResponse)
async def recommend_products_detailed(request: RecommendationRequest):
    """
    Get detailed product recommendations with scores and descriptions.
    
    Args:
        request: Request containing search query text
        
    Returns:
        Detailed list of recommended products with similarity scores
        
    Raises:
        HTTPException: If recommendation search fails
    """
    try:
        logger.info(f"Received detailed recommendation request: {request.text}")
        
        # Check if vector store is available
        if not vector_store.is_connected():
            logger.warning("Qdrant vector store not available")
            raise HTTPException(
                status_code=503,
                detail="Product recommendation service is unavailable. Qdrant server may not be running."
            )
        
        # Ensure collection exists and has data
        try:
            collection_info = vector_store.get_collection_info()
            if collection_info is None or collection_info.get('points_count', 0) == 0:
                logger.info("Vector store collection is empty, adding sample products...")
                vector_store.add_sample_products()
        except Exception as e:
            logger.warning(f"Failed to check collection, creating new one: {e}")
            vector_store.create_collection()
            vector_store.add_sample_products()
        
        # Search for similar products
        similar_products = vector_store.search_similar_products(
            query=request.text,
            limit=5  # Get more results for detailed view
        )
        
        if not similar_products:
            logger.info("No similar products found")
            return DetailedRecommendationResponse(recommendations=[])
        
        # Format detailed recommendations - FIXED: Handle potential missing keys
        recommendations = []
        for product in similar_products:
            recommendations.append(
                DetailedRecommendationResponse.RecommendationItem(
                    product_id=str(product.get("id", "")),
                    product_title=product.get("product_title", ""),
                    product_description=product.get("product_description", ""),
                    similarity_score=round(product.get("score", 0.0), 4)
                )
            )
        
        logger.info(f"Found {len(recommendations)} detailed product recommendations")
        
        return DetailedRecommendationResponse(recommendations=recommendations)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detailed recommendation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during detailed product recommendation"
        )


@router.get("/vector-store/health")
async def get_vector_store_health():
    """
    Health check for vector store.
    
    Returns:
        Vector store health status
    """
    try:
        is_connected = vector_store.is_connected()
        collection_info = vector_store.get_collection_info() if is_connected else None
        
        return {
            "status": "connected" if is_connected else "disconnected",
            "collection_info": collection_info,
            "message": "Vector store is ready" if is_connected else "Vector store is not available"
        }
        
    except Exception as e:
        logger.error(f"Error checking vector store health: {str(e)}")
        return {
            "status": "error",
            "collection_info": None,
            "message": f"Error checking vector store: {str(e)}"
        }


@router.get("/vector-store/info")
async def get_vector_store_info():
    """
    Get information about the vector store.
    
    Returns:
        Vector store status and collection information
    """
    try:
        if not vector_store.is_connected():
            return {
                "status": "disconnected",
                "message": "Qdrant server is not available",
                "collection_info": None
            }
        
        collection_info = vector_store.get_collection_info()
        
        return {
            "status": "connected",
            "message": "Vector store is available",
            "collection_info": collection_info,
            "qdrant_host": vector_store.qdrant_host,
            "qdrant_port": vector_store.qdrant_port,
            "collection_name": vector_store.collection_name
        }
        
    except Exception as e:
        logger.error(f"Error getting vector store info: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving vector store information: {str(e)}",
            "collection_info": None
        }


@router.post("/vector-store/setup")
async def setup_vector_store():
    """
    Setup the vector store with sample products.
    
    Returns:
        Setup status and information
    """
    try:
        if not vector_store.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Qdrant server is not available"
            )
        
        # Create collection if it doesn't exist
        vector_store.create_collection()
        
        # Add sample products
        vector_store.add_sample_products()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return {
            "message": "Vector store setup completed successfully",
            "collection_info": collection_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error setting up vector store: {str(e)}"
        )

@router.get("/vector-store/health")
async def vector_store_health():
    """Detailed health check for vector store."""
    try:
        if not vector_store.is_connected():
            return {
                "status": "disconnected",
                "message": "Cannot connect to Qdrant server",
                "host": vector_store.qdrant_host,
                "port": vector_store.qdrant_port
            }
        
        # Test basic operations
        collection_info = vector_store.get_collection_info()
        
        return {
            "status": "healthy",
            "message": "Vector store is connected and operational",
            "host": vector_store.qdrant_host,
            "port": vector_store.qdrant_port,
            "collection_info": collection_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Vector store health check failed: {str(e)}",
            "host": vector_store.qdrant_host,
            "port": vector_store.qdrant_port
        }
    
@router.post("/vector-store/reset")
async def reset_vector_store():
    """
    Reset the vector store (delete and recreate with sample data).
    
    Returns:
        Reset status and information
    """
    try:
        if not vector_store.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Qdrant server is not available"
            )
        
        # Recreate collection
        vector_store.create_collection(recreate=True)
        
        # Add sample products
        vector_store.add_sample_products()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return {
            "message": "Vector store reset completed successfully",
            "collection_info": collection_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting vector store: {str(e)}"
        )


@router.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information
    """
    return {
        "message": "Sentiment Analysis & Product Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "predict": "POST /predict - Single text sentiment prediction",
            "batch_predict": "POST /predict/batch - Batch text sentiment prediction",
            "recommend": "POST /recommend - Product recommendations",
            "recommend_detailed": "POST /recommend/detailed - Detailed product recommendations",
            "model_info": "GET /model/info - Model information",
            "vector_store_health": "GET /vector-store/health - Vector store health check",
            "vector_store_info": "GET /vector-store/info - Vector store information",
            "vector_store_setup": "POST /vector-store/setup - Setup vector store with sample data",
            "vector_store_reset": "POST /vector-store/reset - Reset vector store"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }