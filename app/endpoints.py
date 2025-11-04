"""
API endpoints for the sentiment analysis service.

This module contains the route handlers for the FastAPI application.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Optional

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
        
        # FIX: Actually call the classifier instead of using placeholder values
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
            detail=f"Internal server error during prediction: {str(e)}"
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
        
        # Make batch predictions
        results = clf.predict_batch(request.texts)
        
        # Convert to response format
        predictions = [
            PredictionResponse(label=label, confidence=confidence)
            for label, confidence in results
        ]
        
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
            detail=f"Internal server error during batch prediction: {str(e)}"
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
            detail=f"Error retrieving model information: {str(e)}"
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
        
        # Search for similar products
        similar_products = vector_store.search_similar_products(
            query=request.text,
            limit=3
        )
        
        if not similar_products:
            logger.info("No similar products found")
            return RecommendationResponse(recommended_products=[])
        
        # Extract product titles
        product_names = [product["product_title"] for product in similar_products]
        
        logger.info(f"Found {len(product_names)} product recommendations")
        
        return RecommendationResponse(recommended_products=product_names)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during product recommendation: {str(e)}"
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
        
        # Search for similar products
        similar_products = vector_store.search_similar_products(
            query=request.text,
            limit=5  # Get more results for detailed view
        )
        
        if not similar_products:
            logger.info("No similar products found")
            return DetailedRecommendationResponse(recommendations=[])
        
        # Format detailed recommendations
        recommendations = []
        for product in similar_products:
            recommendations.append(
                DetailedRecommendationResponse.RecommendationItem(
                    product_id=product["id"],
                    product_title=product["product_title"],
                    product_description=product["product_description"],
                    similarity_score=round(product["score"], 4)
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
            detail=f"Internal server error during detailed product recommendation: {str(e)}"
        )


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


@router.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information
    """
    return {
        "message": "Sentiment Analysis & Product Recommendation API",  # FIX: Corrected API name
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Single text sentiment prediction",
            "batch_predict": "POST /predict/batch - Batch text sentiment prediction",
            "recommend": "POST /recommend - Product recommendations",
            "recommend_detailed": "POST /recommend/detailed - Detailed product recommendations",
            "health": "GET /health - Health check",
            "model_info": "GET /model/info - Model information",
            "vector_store_info": "GET /vector-store/info - Vector store information",
            "vector_store_setup": "POST /vector-store/setup - Setup vector store with sample data"
        }
    }