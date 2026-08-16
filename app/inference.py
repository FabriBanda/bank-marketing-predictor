# como se consulta el modelo

from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd

from app.schemas import PredictionRequest

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "bank_marketing_pipeline.joblib"
)

# se usa para que solo en la primera peticion se cargue el modelo, y en las siguientes se reutiliza el mismo pipeline en memoria
@lru_cache() # guarda el resultado de la primera ejecución
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo"
        )

# recupera el modelo ya entrenado con su StandardScaler, su OneHotEncoder y la LogisticRegression incluidos en el pipeline.
    return joblib.load(MODEL_PATH)


def make_prediction(data: PredictionRequest) -> tuple[str, float]:

    # convertir los datos a una tabla de una sola fila m de JSON A DataFrame, para que el pipeline pueda procesarlos
    row = pd.DataFrame([data.model_dump()])

    # recuperamos el pipeline y pedimos la clase predicha
    model = load_model()
    prediction = str(model.predict(row)[0])

    # encontramos probabilidad de que la prediccion sea "yes"
    yes_index = list(model.classes_).index("yes")

    probability_yes = float(model.predict_proba(row)[0][yes_index])

    return prediction, probability_yes


