#!/usr/bin/env python3
"""
Test runner script for BERT Sentiment Analysis API.
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path


def start_api_server():
    """Start the API server in a subprocess."""
    print("🚀 Starting API server...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app", 
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    max_wait = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print(" API server started successfully")
                return process
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass
        
        print("⏳ Waiting for server to start...")
        time.sleep(2)
    
    print(" Failed to start API server")
    process.terminate()
    return None


def run_tests():
    """Run the test suite."""
    print("\nRunning tests...")
    
    # Run pytest with specific options
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "test_api.py", 
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    return result.returncode


def main():
    """Main function."""
    print("BERT Sentiment Analysis API Test Suite")
    print("=" * 50)
    
    # Start server
    server_process = start_api_server()
    if not server_process:
        sys.exit(1)
    
    try:
        # Run tests
        return_code = run_tests()
        
        # Return the test result
        sys.exit(return_code)
        
    finally:
        # Stop server
        if server_process:
            print("\n Stopping API server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()