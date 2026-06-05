from fastapi import FastAPI

from .inference import IntentPredictor
from .schemas import PredictRequest, PredictResponse


app = FastAPI(title="Smart Support Router")
predictor = IntentPredictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    return predictor.predict(request.text)
