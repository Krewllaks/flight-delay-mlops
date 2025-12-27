import time
import requests
import sys

# URL of the service to test
URL = "http://localhost:80/predict"
HEALTH_URL = "http://localhost:80/"

def smoke_test():
    """
    Waits for the service to be ready and then sends a test request.
    This simulates a deployment verification test.
    """
    print("Beginning smoke test...")
    
    # Retry logic to wait for container to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            health = requests.get(HEALTH_URL)
            if health.status_code == 200:
                print("Health check passed!")
                break
        except requests.ConnectionError:
            print(f"Waiting for service... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("Service failed to start or is not reachable.")
        sys.exit(1)

    # Test prediction endpoint
    payload = {
        "airport_code": "AMS",
        "delay_minutes": 10.0
    }
    
    try:
        print(f"Sending test request to {URL}...")
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
             print("Smoke test passed! Response:", response.json())
             sys.exit(0)
        else:
             print(f"Smoke test failed. Status: {response.status_code}, Body: {response.text}")
             sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()
