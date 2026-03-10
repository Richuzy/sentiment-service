# api/tests/test_api.py
from fastapi.testclient import TestClient
import ml.inference as inference_module

class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, text):
        return {"label": "Positive", "confidence": 0.99}

def test_health():
    from api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict(monkeypatch):
    import api.main as api_main

    # Patch the SentimentModel reference used inside api.main
    monkeypatch.setattr(api_main, "SentimentModel", DummyModel)

    # Reset global model and run startup manually
    api_main.model = None
    api_main.load_model_on_startup()

    client = TestClient(api_main.app)
    res = client.post("/predict", json={"text": "lalettan polichu"})
    assert res.status_code == 200
    assert res.json()["sentiment"] == "Positive"