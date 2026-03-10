from fastapi import FastAPI 
from pydantic import BaseModel
from ml.inference import SentimentModel

app = FastAPI(
    title = "Malayalam Sentiment API",
    version = "1.0.0"
)

#load model once after startup

model= SentimentModel("model")
class PredicRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    
@app.get ("/health")
def health():
    return {"status": "ok"}    
        
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredicRequest):
    result = model.predict(request.text)
    return PredictResponse(
        sentiment = result["label"],
        confidence = result['confidence']
    )
        
        