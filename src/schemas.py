from pydantic import BaseModel


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
