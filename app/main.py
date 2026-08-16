from fastapi import FastAPI

from app.inference import make_prediction
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Bank Marketing Predictor",
    description="API para estimar la probabilidad de contratación de un depósito a plazo.",
    version="1.0.0",
)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> PredictionResponse:

    prediction, probability = make_prediction(data)

    if prediction == "yes":
        message = "Potencialmente interesado"
    else:
        message = "Probablemente no interesado"

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        message=message,
    )

from fastapi import FastAPI

from app.inference import make_prediction
from app.schemas import PredictionRequest, PredictionResponse

