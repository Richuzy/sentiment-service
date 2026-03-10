# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml.inference import SentimentModel

app = FastAPI(title="Malayalam Sentiment API", version="1.0.0")

model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float


@app.on_event("startup")
def load_model_on_startup():
    global model

    # ✅ Skip model loading during tests/CI
    if os.getenv("SKIP_MODEL_LOAD", "0") == "1":
        print("Skipping model load (SKIP_MODEL_LOAD=1)")
        model = None
        return

    model_dir = os.getenv("MODEL_DIR", "model")

    try:
        model = SentimentModel(model_dir)
        print("Model loaded successfully.")
    except Exception as e:
        model = None
        print(f"Warning: model failed to load at startup: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = model.predict(request.text)

    return PredictResponse(
        sentiment=result["label"],
        confidence=result["confidence"],
    )