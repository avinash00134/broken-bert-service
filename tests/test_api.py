"""
Test cases for the BERT Sentiment Analysis API.
"""

import pytest
import requests
import json
import pandas as pd
from pathlib import Path


class TestSentimentAPI:
    """Test cases for sentiment analysis API endpoints."""
    
    def test_health_endpoint(self, api_server):
        """Test health check endpoint."""
        response = requests.get(f"{api_server}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_predict_single_text(self, api_server, sample_texts):
        """Test single text prediction endpoint."""
        # Test positive text
        positive_text = sample_texts["positive"][0]
        response = requests.post(
            f"{api_server}/predict",
            json={"text": positive_text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["prediction"] in ["positive", "negative"]
        assert 0 <= data["confidence"] <= 1
        
        # Test negative text
        negative_text = sample_texts["negative"][0]
        response = requests.post(
            f"{api_server}/predict",
            json={"text": negative_text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
    
    def test_predict_batch_texts(self, api_server, sample_texts):
        """Test batch text prediction endpoint."""
        # Mix of positive and negative texts
        test_texts = sample_texts["positive"][:2] + sample_texts["negative"][:2]
        
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": test_texts}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(test_texts)
        
        for prediction in data["predictions"]:
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert prediction["prediction"] in ["positive", "negative"]
            assert 0 <= prediction["confidence"] <= 1
    
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
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_invalid_json(self, api_server):
        """Test prediction with invalid JSON."""
        response = requests.post(
            f"{api_server}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_batch_empty_list(self, api_server):
        """Test batch prediction with empty list."""
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["predictions"] == []
    
    def test_model_info_endpoint(self, api_server):
        """Test model information endpoint."""
        response = requests.get(f"{api_server}/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "n_classes" in data
        assert "labels" in data
        assert "device" in data
        assert data["model_type"] == "DistilBERT"
        assert data["n_classes"] == 2
        assert set(data["labels"]) == {"negative", "positive"}


class TestTrainingAPI:
    """Test cases for model training API endpoints."""
    
    def test_train_model_endpoint(self, api_server, test_data_dir):
        """Test model training endpoint."""
        # Create test training data
        test_data = [
            {"review": "This product is amazing!", "label": "positive"},
            {"review": "Terrible quality, very disappointed", "label": "negative"},
            {"review": "Great value for money", "label": "positive"},
            {"review": "Poor customer service", "label": "negative"},
            {"review": "Excellent product quality", "label": "positive"},
            {"review": "Waste of money", "label": "negative"},
        ]
        
        test_df = pd.DataFrame(test_data)
        test_csv_path = test_data_dir / "test_training.csv"
        test_df.to_csv(test_csv_path, index=False)
        
        training_config = {
            "csv_path": str(test_csv_path),
            "model_save_path": "assets/test_model.pth",
            "tokenizer_save_path": "assets/test_tokenizer",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "max_length": 32
        }
        
        response = requests.post(
            f"{api_server}/train",
            json=training_config
        )
        
        # Training might take time, so accept 200 or 202
        assert response.status_code in [200, 202]
        data = response.json()
        
        if response.status_code == 200:
            assert "history" in data
            assert "train_accuracy" in data["history"]
            assert "test_accuracy" in data["history"]
        else:  # 202 Accepted
            assert "message" in data
            assert "training_started" in data
    
    def test_train_model_invalid_csv(self, api_server):
        """Test training with invalid CSV path."""
        training_config = {
            "csv_path": "nonexistent.csv",
            "model_save_path": "assets/test_model.pth",
            "tokenizer_save_path": "assets/test_tokenizer",
            "epochs": 1
        }
        
        response = requests.post(
            f"{api_server}/train",
            json=training_config
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_train_model_missing_required_fields(self, api_server):
        """Test training with missing required fields."""
        # Missing csv_path
        training_config = {
            "model_save_path": "assets/test_model.pth",
            "epochs": 1
        }
        
        response = requests.post(
            f"{api_server}/train",
            json=training_config
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestModelManagementAPI:
    """Test cases for model management API endpoints."""
    
    def test_reload_model_endpoint(self, api_server):
        """Test model reload endpoint."""
        response = requests.post(f"{api_server}/model/reload")
        # Should return 200 if reload successful, or 400 if no model loaded
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "model_info" in data
        else:
            data = response.json()
            assert "error" in data
    
    def test_save_model_endpoint(self, api_server):
        """Test model save endpoint."""
        save_config = {
            "model_path": "assets/test_save_model.pth",
            "tokenizer_path": "assets/test_tokenizer"
        }
        
        response = requests.post(
            f"{api_server}/model/save",
            json=save_config
        )
        
        # Could be 200 (success) or 400 (no model to save)
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "model_path" in data
            assert "tokenizer_path" in data
        else:
            data = response.json()
            assert "error" in data


class TestPerformanceAPI:
    """Test cases for API performance."""
    
    def test_response_time_single_prediction(self, api_server, sample_texts):
        """Test response time for single prediction."""
        import time
        
        start_time = time.time()
        response = requests.post(
            f"{api_server}/predict",
            json={"text": sample_texts["positive"][0]}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should respond within 5 seconds
        assert response_time < 5.0
    
    def test_response_time_batch_prediction(self, api_server, sample_texts):
        """Test response time for batch prediction."""
        import time
        
        # Create batch of 10 texts
        test_texts = sample_texts["positive"][:3] + sample_texts["negative"][:3] + sample_texts["neutral"][:4]
        
        start_time = time.time()
        response = requests.post(
            f"{api_server}/predict/batch",
            json={"texts": test_texts}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Batch prediction should be reasonable
        assert response_time < 10.0


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_nonexistent_endpoint(self, api_server):
        """Test request to nonexistent endpoint."""
        response = requests.get(f"{api_server}/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_http_method(self, api_server):
        """Test invalid HTTP method for endpoints."""
        # GET instead of POST for predict
        response = requests.get(f"{api_server}/predict")
        assert response.status_code == 405
    
    def test_large_text_input(self, api_server):
        """Test prediction with very large text input."""
        large_text = "This is a test. " * 1000  # Very large text
        
        response = requests.post(
            f"{api_server}/predict",
            json={"text": large_text}
        )
        
        # Should handle large texts gracefully (either process or return error)
        assert response.status_code in [200, 400, 413]
    
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
                json={"text": text}
            )
            assert response.status_code in [200, 400]
            
            if response.status_code == 200:
                data = response.json()
                assert "prediction" in data
                assert "confidence" in data


def test_concurrent_requests(api_server, sample_texts):
    """Test handling concurrent requests."""
    import threading
    import time
    
    results = []
    errors = []
    
    def make_prediction(text, result_list, error_list):
        try:
            response = requests.post(
                f"{api_server}/predict",
                json={"text": text},
                timeout=10
            )
            result_list.append(response.status_code)
        except Exception as e:
            error_list.append(str(e))
    
    threads = []
    test_texts = sample_texts["positive"][:2] + sample_texts["negative"][:2]
    
    for text in test_texts:
        thread = threading.Thread(
            target=make_prediction,
            args=(text, results, errors)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=15)
    
    # Check that we got responses (allow some failures under load)
    assert len(results) + len(errors) == len(test_texts)
    if results:
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])