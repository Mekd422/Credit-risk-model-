from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "feature_1": 123,
        "feature_2": 0.5,

    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "risk_probability" in response.json()
