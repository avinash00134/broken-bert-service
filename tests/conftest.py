"""
Pytest configuration and shared fixtures.
"""

import pytest
import requests
import time
import os
from pathlib import Path


@pytest.fixture(scope="session")
def api_server():
    """Fixture to provide the API server URL with health check."""
    base_url = "http://localhost:8000"
    
    # Wait for server to be ready (max 30 seconds)
    max_wait = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f" API server is ready at {base_url}")
                return base_url
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass
        
        print("⏳ Waiting for API server to start...")
        time.sleep(2)
    
    pytest.skip("API server is not available")


@pytest.fixture(scope="session")
def sample_texts():
    """Fixture to provide sample texts for testing."""
    return {
        "positive": [
            "This movie was absolutely fantastic! Great acting and wonderful storyline.",
            "I love this product! It exceeded all my expectations.",
            "Excellent quality and amazing value for the price.",
            "Outstanding service and wonderful experience overall.",
            "Highly recommended! This is exactly what I was looking for."
        ],
        "negative": [
            "This was terrible! Waste of time and money.",
            "Poor quality and disappointing performance.",
            "I would not recommend this to anyone. Very bad experience.",
            "Absolutely horrible! The worst I've ever seen.",
            "Low quality materials and terrible customer service."
        ],
        "neutral": [
            "The product arrived on time and works as expected.",
            "It's okay, nothing special but gets the job done.",
            "Average quality for the price point.",
            "Meets basic requirements but has room for improvement."
        ]
    }


@pytest.fixture
def test_data_dir(tmp_path_factory):
    """Fixture to provide temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session", autouse=True)
def check_api_dependencies():
    """Check if API dependencies are available before running tests."""
    try:
        import transformers
        import torch
        import qdrant_client
    except ImportError as e:
        pytest.skip(f"Required dependency not available: {e}")


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "vector_store: mark test as vector store test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers and conditions."""
    skip_slow = pytest.mark.skip(reason="slow test - use --run-slow to run")
    skip_integration = pytest.mark.skip(reason="integration test - use --run-integration to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)