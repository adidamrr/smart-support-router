from fastapi import FastAPI

from .database import get_recent_predictions, init_db, save_prediction
from .inference import IntentPredictor
from .schemas import PredictRequest, PredictResponse, PredictionHistoryItem


app = FastAPI(title="Smart Support Router")
predictor = IntentPredictor()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    result = predictor.predict(request.text)
    save_prediction(
        text=result["text"],
        predicted_intent=result["intent"],
        confidence=result["confidence"],
        suggested_reply=result["suggested_reply"],
    )
    return result


@app.get("/history", response_model=list[PredictionHistoryItem])
def history(limit: int = 20) -> list:
    return get_recent_predictions(limit)
