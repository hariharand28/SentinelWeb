from pydantic import BaseModel, HttpUrl


class PredictionRequest(BaseModel):
    url: HttpUrl


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float