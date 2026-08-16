from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.inference import make_prediction
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Bank Marketing Predictor",
    description="API para estimar la probabilidad de contratación de un depósito a plazo.",
    version="1.0.0",
)

# Publicamos la interfaz desde la misma API para que el formulario y /predict
# compartan el mismo origen y el navegador no bloquee la solicitud por CORS.
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

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
