"""
Test cases for the BERT Sentiment Analysis API.
"""

import pytest
import requests
import json
import pandas as pd
from pathlib import Path
import time
import threading


class TestSentimentAPI:
    """Test cases for sentiment analysis API endpoints."""
    
    def test_health_endpoint(self, api_server):
        """Test health check endpoint."""
        response = requests.get(f"{api_server}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_ready" in data
        assert "message" in data
    
    def test_predict_single_text(self, api_server, sample_texts):
        """Test single text prediction endpoint."""
        # Test positive text
        positive_text = sample_texts["positive"][0]
        response = requests.post(
            f"{api_server}/predict",
            json={"text": positive_text}
        )
        
        # If model is loaded, should return 200, otherwise 503
        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert data["label"] in ["positive", "negative"]
            assert 0 <= data["confidence"] <= 1
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data
            assert "model" in data["detail"].lower()
    
    def test_predict_batch_texts(self, api_server, sample_texts):
        """Test batch text prediction endpoint."""
        # Mix of positive and negative texts
        test_texts = sample_texts["positive"][:2] + sample_texts["negative"][:2]
        
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": test_texts}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == len(test_texts)
            
            for prediction in data["predictions"]:
                assert "label" in prediction
                assert "confidence" in prediction
                assert prediction["label"] in ["positive", "negative"]
                assert 0 <= prediction["confidence"] <= 1
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data
            assert "model" in data["detail"].lower()
    
    def test_predict_empty_text(self, api_server):
        """Test prediction with empty text."""
        response = requests.post(
            f"{api_server}/predict",
            json={"text": ""}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_missing_field(self, api_server):
        """Test prediction with missing required field."""
        response = requests.post(
            f"{api_server}/predict",
            json={}  # Missing 'text' field
        )
        assert response.status_code == 422  # FastAPI returns 422 for validation errors
    
    def test_predict_invalid_json(self, api_server):
        """Test prediction with invalid JSON."""
        response = requests.post(
            f"{api_server}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_batch_empty_list(self, api_server):
        """Test batch prediction with empty list."""
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422  # Validation error for empty list
    
    def test_model_info_endpoint(self, api_server):
        """Test model information endpoint."""
        response = requests.get(f"{api_server}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "n_classes" in data
            assert "labels" in data
            assert "device" in data
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data


class TestVectorStoreAPI:
    """Test cases for vector store and recommendation endpoints."""
    
    def test_vector_store_info_endpoint(self, api_server):
        """Test vector store information endpoint."""
        response = requests.get(f"{api_server}/vector-store/info")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "collection_info" in data
    
    def test_vector_store_setup_endpoint(self, api_server):
        """Test vector store setup endpoint."""
        response = requests.post(f"{api_server}/vector-store/setup")
        
        # Could be 200 (success), 503 (Qdrant not available), or 500 (other error)
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "collection_info" in data
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data
            assert "Qdrant" in data["detail"]
    
    def test_recommend_products_endpoint(self, api_server):
        """Test product recommendation endpoint."""
        test_queries = [
            "fast laptop for programming",
            "wireless headphones",
            "gaming computer",
            "monitor for work"
        ]
        
        for query in test_queries:
            response = requests.post(
                f"{api_server}/recommend",
                json={"text": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "recommended_products" in data
                assert isinstance(data["recommended_products"], list)
                # Could be empty list if no products found
            elif response.status_code == 503:
                data = response.json()
                assert "detail" in data
                assert "Qdrant" in data["detail"] or "unavailable" in data["detail"]
    
    def test_detailed_recommendations_endpoint(self, api_server):
        """Test detailed product recommendations endpoint."""
        response = requests.post(
            f"{api_server}/recommend/detailed",
            json={"text": "laptop for development work"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
            
            if data["recommendations"]:
                recommendation = data["recommendations"][0]
                assert "product_id" in recommendation
                assert "product_title" in recommendation
                assert "product_description" in recommendation
                assert "similarity_score" in recommendation
                assert 0 <= recommendation["similarity_score"] <= 1
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data
    
    def test_recommend_empty_query(self, api_server):
        """Test recommendation with empty query."""
        response = requests.post(
            f"{api_server}/recommend",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_recommend_missing_field(self, api_server):
        """Test recommendation with missing field."""
        response = requests.post(
            f"{api_server}/recommend",
            json={}  # Missing 'text' field
        )
        assert response.status_code == 422  # Validation error


class TestRootEndpoints:
    """Test cases for root and information endpoints."""
    
    def test_root_endpoint(self, api_server):
        """Test root endpoint."""
        response = requests.get(f"{api_server}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)
    
    def test_api_documentation(self, api_server):
        """Test API documentation endpoints."""
        # Test Swagger UI
        response = requests.get(f"{api_server}/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = requests.get(f"{api_server}/redoc")
        assert response.status_code == 200


class TestPerformanceAPI:
    """Test cases for API performance."""
    
    def test_response_time_single_prediction(self, api_server, sample_texts):
        """Test response time for single prediction."""
        start_time = time.time()
        response = requests.post(
            f"{api_server}/predict",
            json={"text": sample_texts["positive"][0]},
            timeout=30
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within reasonable time
        assert response_time < 30.0
        
        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "confidence" in data
    
    def test_response_time_batch_prediction(self, api_server, sample_texts):
        """Test response time for batch prediction."""
        # Create batch of texts
        test_texts = sample_texts["positive"][:3] + sample_texts["negative"][:3]
        
        start_time = time.time()
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": test_texts},
            timeout=30
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Batch prediction should be reasonable
        assert response_time < 30.0
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == len(test_texts)
    
    def test_response_time_recommendations(self, api_server):
        """Test response time for recommendations."""
        start_time = time.time()
        response = requests.post(
            f"{api_server}/recommend",
            json={"text": "laptop computer"},
            timeout=30
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within reasonable time
        assert response_time < 30.0


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_nonexistent_endpoint(self, api_server):
        """Test request to nonexistent endpoint."""
        response = requests.get(f"{api_server}/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_invalid_http_method(self, api_server):
        """Test invalid HTTP method for endpoints."""
        # GET instead of POST for predict
        response = requests.get(f"{api_server}/predict")
        assert response.status_code == 405  # Method Not Allowed
    
    def test_large_text_input(self, api_server):
        """Test prediction with very large text input."""
        large_text = "This is a test. " * 1000  # Very large text
        
        response = requests.post(
            f"{api_server}/predict",
            json={"text": large_text},
            timeout=30
        )
        
        # Should handle large texts gracefully
        assert response.status_code in [200, 400, 422, 503]
    
    def test_special_characters_text(self, api_server):
        """Test prediction with special characters."""
        special_texts = [
            "Text with emoji 😊 and symbols #@$%",
            "Unicode text: 中文 Español Français",
            "Text with <html> tags & symbols",
            "Mixed: Hello 世界! 😊 #test"
        ]
        
        for text in special_texts:
            response = requests.post(
                f"{api_server}/predict",
                json={"text": text},
                timeout=10
            )
            assert response.status_code in [200, 400, 422, 503]
    
    def test_malformed_requests(self, api_server):
        """Test various malformed requests."""
        # No Content-Type header
        response = requests.post(f"{api_server}/predict", data="some data")
        assert response.status_code in [400, 415, 422]
        
        # Wrong Content-Type
        response = requests.post(
            f"{api_server}/predict",
            data='{"text": "test"}',
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [400, 415, 422]


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    def test_concurrent_predictions(self, api_server, sample_texts):
        """Test handling concurrent prediction requests."""
        results = []
        errors = []
        
        def make_prediction(text, result_list, error_list):
            try:
                response = requests.post(
                    f"{api_server}/predict",
                    json={"text": text},
                    timeout=10
                )
                result_list.append({
                    "status_code": response.status_code,
                    "text": text
                })
            except Exception as e:
                error_list.append(str(e))
        
        threads = []
        test_texts = sample_texts["positive"][:3] + sample_texts["negative"][:3]
        
        for text in test_texts:
            thread = threading.Thread(
                target=make_prediction,
                args=(text, results, errors)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=15)
        
        # Check that we got responses
        assert len(results) + len(errors) == len(test_texts)
        
        # Log results for debugging
        print(f"Successful: {len(results)}, Errors: {len(errors)}")
        
        # If we have successful responses, check they're valid
        for result in results:
            if result["status_code"] == 200:
                # We can't easily validate the content without making another request
                pass


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_text_length_boundaries(self, api_server):
        """Test text length boundaries."""
        # Very short text
        short_text = "a"
        response = requests.post(
            f"{api_server}/predict",
            json={"text": short_text}
        )
        assert response.status_code in [200, 400, 422, 503]
        
        # Long but acceptable text
        long_text = "This is a test. " * 100  # ~1600 characters
        response = requests.post(
            f"{api_server}/predict",
            json={"text": long_text}
        )
        assert response.status_code in [200, 400, 422, 503]
    
    def test_batch_size_limits(self, api_server, sample_texts):
        """Test batch size limits."""
        # Large batch
        large_batch = sample_texts["positive"] * 10  # Create large batch
        
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": large_batch},
            timeout=60
        )
        
        # Should handle large batches or return appropriate error
        assert response.status_code in [200, 400, 413, 422, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == len(large_batch)


def test_complete_workflow(api_server, sample_texts):
    """Test complete workflow including vector store setup and recommendations."""
    # 1. Check health
    health_response = requests.get(f"{api_server}/health")
    assert health_response.status_code == 200
    
    # 2. Setup vector store if available
    setup_response = requests.post(f"{api_server}/vector-store/setup")
    if setup_response.status_code == 200:
        # 3. Test recommendations
        recommend_response = requests.post(
            f"{api_server}/recommend",
            json={"text": "laptop for programming"}
        )
        assert recommend_response.status_code in [200, 503]
        
        # 4. Test detailed recommendations
        detailed_response = requests.post(
            f"{api_server}/recommend/detailed",
            json={"text": "wireless headphones"}
        )
        assert detailed_response.status_code in [200, 503]
    
    # 5. Test sentiment analysis if model is loaded
    predict_response = requests.post(
        f"{api_server}/predict",
        json={"text": sample_texts["positive"][0]}
    )
    assert predict_response.status_code in [200, 503]
    
    # 6. Get vector store info
    vector_info_response = requests.get(f"{api_server}/vector-store/info")
    assert vector_info_response.status_code == 200


# Fixtures for pytest
@pytest.fixture
def api_server():
    """Fixture to provide the API server URL."""
    return "http://localhost:8000"


@pytest.fixture
def sample_texts():
    """Fixture to provide sample texts for testing."""
    return {
        "positive": [
            "This product is absolutely amazing! I love it!",
            "Excellent quality and great value for money.",
            "Outstanding performance and beautiful design.",
            "Highly recommended! Will buy again for sure.",
            "Fantastic product with excellent customer service."
        ],
        "negative": [
            "Terrible product, very disappointed with the quality.",
            "Poor craftsmanship and bad customer service.",
            "Waste of money, would not recommend to anyone.",
            "Absolutely horrible experience with this product.",
            "Low quality materials and poor performance."
        ],
        "neutral": [
            "The product arrived on time and as described.",
            "It's okay for the price, nothing special.",
            "Average product with standard features.",
            "Meets basic expectations but not exceptional."
        ]
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Fixture to provide temporary directory for test data."""
    return tmp_path


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])