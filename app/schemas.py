#Este archivo su idea es definir el "contrato" de los daros que recibirá la API, para que FastAPI pueda validar que los datos que le llegan son correctos

from typing import Literal

# libreria que fastApi usa para validar datos
from pydantic import BaseModel, Field

# PredictionRequest: nombre de los datos que recibirá /predict
class PredictionRequest(BaseModel):

    # Recibira TODAS LAS VARIABLES DE ENTRADA QUE USAMOS PARA ENTRENAR EL MODELO
    # ge = “greater than or equal”
    # le = ““less than or equal”:”
    age:int = Field(ge=18,le=100, description="Edad del cliente, debe ser un entero entre 18 y 100")
    balance:float = Field(ge=0, description="Balance del cliente, debe ser un numero positivo")
    campaign:int = Field(ge=1, description="Numero de contactos realizados durante la campaña, debe ser un entero positivo")
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]

class PredictionResponse(BaseModel):
    # Recibira la prediccion del modelo, que sera un string "yes" o "no"
    prediction: Literal["yes", "no"]
    probability: float
    message: Literal[
        "Potencialmente interesado",
        "Probablemente no interesado",
    ]