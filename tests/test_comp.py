from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    request_data = {
        "airport_code": "JFK",
        "delay_minutes": 50.0
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "airport_bucket" in data
    assert "delay_category" in data
    assert data["airport_code"] == "JFK"
    assert data["delay_category"] == "long_delay"

def test_predict_endpoint_short_delay():
    request_data = {
        "airport_code": "IST",
        "delay_minutes": 20.0
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["delay_category"] == "short_delay"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Flight Delay Prediction Service is running"}
