# api/tests/test_api.py
import os
from fastapi.testclient import TestClient

# Always skip loading the real model during tests
os.environ["SKIP_MODEL_LOAD"] = "1"


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, text: str):
        return {"label": "Positive", "confidence": 0.99}


def test_health():
    import api.main as api_main

    client = TestClient(api_main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(monkeypatch):
    import api.main as api_main

    # Patch the SentimentModel reference used by api.main
    monkeypatch.setattr(api_main, "SentimentModel", DummyModel)

    # Temporarily allow startup to run, but it will instantiate DummyModel
    os.environ["SKIP_MODEL_LOAD"] = "0"
    api_main.model = None
    api_main.load_model_on_startup()
    os.environ["SKIP_MODEL_LOAD"] = "1"

    client = TestClient(api_main.app)
    res = client.post("/predict", json={"text": "lalettan polichu"})

    assert res.status_code == 200
    body = res.json()
    assert body["sentiment"] == "Positive"
    assert 0.0 <= body["confidence"] <= 1.0