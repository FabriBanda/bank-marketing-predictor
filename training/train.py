from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/bank.csv")
MODEL_PATH = Path("models/bank_marketing_pipeline.joblib")
# hazlo siempre con la misma mezcla aleatoria
RANDOM_STATE = 42
# Usare el 20% de los datos para testear el modelo, y el 80% restante para entrenarlo.
TEST_SIZE = 0.2

# Estas son las columnas que si vamos a usar para entrenar el modelo.
# Se eligieron segun la consigna de la actividad.
FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "campaign",
]
TARGET_COLUMN = "y"

# Aqui separamos las variables numericas de las categoricas
# porque no se preparan igual antes de entrar al modelo.
NUMERIC_FEATURES = ["age", "balance", "campaign"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "housing", "loan"]


def load_dataset() -> pd.DataFrame:
    # Verificamos que el CSV exista en la ruta esperada.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # Este dataset usa punto y coma como separador.
    dataset = pd.read_csv(DATA_PATH, sep=";")

    # Revisamos que todas las columnas necesarias esten presentes.
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]

    # Si falta una columna importante, detenemos el entrenamiento.
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Devolvemos el dataset listo para seguir con el flujo.
    return dataset


def build_pipeline() -> Pipeline:
    # A las columnas numericas les aplicamos escalado
    # para que trabajen en una escala mas comparable.
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # A las categoricas les aplicamos OneHotEncoder
    # para convertir texto en columnas numericas.
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer manda cada grupo de columnas
    # a la transformacion que le corresponde.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # El pipeline junta preprocesamiento y modelo
    # para que siempre se use el mismo flujo en entrenamiento e inferencia.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                # Usamos class_weight="balanced" porque hay muchos mas "no" que "yes".
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    # Probamos el modelo con datos que no vio durante el entrenamiento.
    predictions = model.predict(X_test)

    # Calculamos metricas para entender mejor el comportamiento del modelo.
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="yes", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="yes", zero_division=0),
        "f1_score": f1_score(y_test, predictions, pos_label="yes", zero_division=0),
    }


def main() -> None:
    # Cargamos el dataset completo desde el archivo CSV.
    dataset = load_dataset()

    # X contiene las variables de entrada del modelo.
    X = dataset[FEATURE_COLUMNS]

    # y contiene la respuesta real que queremos aprender a predecir.
    y = dataset[TARGET_COLUMN]

    # Separamos los datos en entrenamiento y prueba.
    # stratify=y mantiene una proporcion parecida de yes y no en ambos grupos.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Construimos el pipeline completo.
    pipeline = build_pipeline()

    # Aqui el modelo aprende a partir de los datos de entrenamiento.
    pipeline.fit(X_train, y_train)

    # Aqui medimos que tan bien funciona con datos no vistos.
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Creamos la carpeta models si todavia no existe.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Guardamos el pipeline entrenado para reutilizarlo despues en la API.
    joblib.dump(pipeline, MODEL_PATH)

    # Imprimimos un resumen del proceso y las metricas obtenidas.
    print("Training completed successfully.")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).round(4).to_dict()}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Model saved to: {MODEL_PATH}")
    print("\nEvaluation metrics:")

    # Recorremos el diccionario de metricas para mostrar cada valor.
    for metric_name, metric_value in metrics.items():
        print(f"- {metric_name}: {metric_value:.4f}")

    # Estas lineas sirven como recordatorio rapido de que significa cada metrica.
    print("\nInterpretation:")
    print("- Accuracy: proportion of total predictions the model got right.")
    print("- Precision: of the clients predicted as interested, how many really said yes.")
    print("- Recall: of the clients who actually said yes, how many the model detected.")
    print("- F1-score: balance between precision and recall.")


if __name__ == "__main__":
    # main() solo se ejecuta si corres este archivo directamente.
    main()
