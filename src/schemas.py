from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    text: str


class TopIntent(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    suggested_reply: str
    top_intents: list[TopIntent]


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    predicted_intent: str
    confidence: float
    suggested_reply: str
    created_at: datetime
