# Bank Marketing Predictor

Proyecto para estimar la probabilidad de que un cliente contrate un depósito a plazo. Se usa una Regresión Logística entrenada con el dataset Bank Marketing de UCI.

La aplicación permite capturar datos básicos del cliente y obtener una predicción mediante una API creada con FastAPI.

## Flujo del proyecto

```text
Frontend -> API /predict -> validación -> pipeline -> modelo -> respuesta -> frontend
```

El frontend solo captura los datos y muestra el resultado. El modelo no se entrena en el frontend ni dentro del endpoint.

## Estructura

```text
data/       Dataset utilizado
training/   Código de entrenamiento
models/     Pipeline entrenado guardado
app/        API, validaciones e inferencia
frontend/   Interfaz web
```

## Variables utilizadas

- `age`: edad.
- `job`: ocupación.
- `marital`: estado civil.
- `education`: nivel educativo.
- `balance`: balance anual promedio.
- `housing`: crédito hipotecario.
- `loan`: préstamo personal.
- `campaign`: número de contactos durante la campaña.

No se utiliza `duration`, ya que la duración de la llamada se conoce después de contactar al cliente. Por eso no estaría disponible al hacer una predicción antes del contacto.

## Instalación y ejecución

Se requiere Python 3.

```bash
pip install -r requirements.txt
```

Para entrenar el modelo y guardar el pipeline:

```bash
python3 training/train.py
```

Para iniciar la API:

```bash
python3 -m uvicorn app.main:app --reload
```

Con la API encendida se puede abrir:

- Documentación y pruebas de la API: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/frontend`

## Entrenamiento y métricas

Se dividieron los datos en 80% para entrenamiento y 20% para prueba. Las variables numéricas se escalan con `StandardScaler` y las categóricas se convierten con `OneHotEncoder`. Ambos procesos forman parte del mismo pipeline que incluye la Regresión Logística.

Métricas obtenidas:

| Métrica | Resultado |
| --- | ---: |
| Accuracy | 0.6336 |
| Precision | 0.1879 |
| Recall | 0.6418 |
| F1-score | 0.2907 |

El accuracy indica que el modelo acertó aproximadamente el 63% de los casos evaluados. El recall es relativamente mayor: detecta una parte importante de los clientes que sí contrataron. Sin embargo, la precision y el F1-score son bajos, lo que indica que varias personas clasificadas como interesadas realmente no contrataron. Esto es esperable porque la clase `yes` aparece menos veces que `no` en el dataset. El objetivo del ejercicio es comprender el flujo de inferencia completo, no conseguir la mejor métrica posible.

## Ejemplo de uso de la API

Solicitud válida a `POST /predict`:

```json
{
  "age": 41,
  "job": "technician",
  "marital": "married",
  "education": "secondary",
  "balance": 3200,
  "housing": "yes",
  "loan": "no",
  "campaign": 2
}
```

Ejemplo de respuesta:

```json
{
  "prediction": "no",
  "probability": 0.3856486994169244,
  "message": "Probablemente no interesado"
}
```

La interfaz muestra esa probabilidad como `38.6%` y cambia el color del resultado según la predicción: verde para `yes` y rojo para `no`.

## Manejo de errores

FastAPI valida los datos con `PredictionRequest` antes de ejecutar el modelo.

Por ejemplo, esta solicitud es inválida porque la edad está fuera del rango permitido:

```json
{
  "age": -10,
  "job": "technician",
  "marital": "married",
  "education": "secondary",
  "balance": 3200,
  "housing": "yes",
  "loan": "no",
  "campaign": 2
}
```

La API responde con el estado `422 Unprocessable Entity`. También ocurre si se manda un tipo incorrecto, por ejemplo `"age": "hola"`.

## Evidencia de funcionamiento

- Caso válido: en `/docs` se puede enviar la solicitud anterior y recibir una predicción y una probabilidad.
- Caso con error: en `/docs` se puede cambiar `age` por `-10` o por `"hola"` y se obtiene una respuesta `422`.
- Frontend: al abrir `/frontend`, llenar el formulario y presionar **Estimar propensión**, la interfaz envía una solicitud real a `/predict` y muestra la respuesta recibida.

Para la entrega se pueden agregar capturas de estos tres casos en esta sección.

## Preguntas de la actividad

### 1. ¿Por qué el modelo se entrena fuera de la API y no dentro de `/predict`?

El entrenamiento puede tardar y no debe repetirse cada vez que un usuario pide una predicción. Por eso se entrena una vez, se guarda el pipeline en un archivo `.joblib` y la API solamente lo carga para usarlo.

### 2. ¿Por qué es importante usar el mismo preprocesamiento en entrenamiento e inferencia?

El modelo aprendió usando datos escalados y variables categóricas convertidas a números. Si en inferencia se hiciera un proceso distinto, las entradas no tendrían el mismo formato y la predicción sería incorrecta o podría fallar. El pipeline evita esa diferencia.

### 3. ¿Qué diferencia existe entre `predict()` y `predict_proba()`?

`predict()` devuelve la clase final, en este caso `yes` o `no`. `predict_proba()` devuelve la probabilidad estimada de cada clase. En este proyecto se obtiene específicamente la probabilidad de `yes`.

### 4. Si el modelo devuelve una probabilidad de 0.72, ¿qué significa y qué no significa?

Significa que, según los patrones aprendidos por el modelo, el cliente tiene una probabilidad estimada de 72% de contratar. No significa que sea una certeza ni que el cliente definitivamente vaya a contratar.

### 5. ¿Por qué `duration` no se debería usar antes de contactar al cliente?

Porque `duration` representa la duración de la llamada y solo se conoce cuando la llamada ya terminó. Usarla antes del contacto sería usar información del futuro y daría una evaluación engañosa del modelo.

### 6. ¿Qué ocurriría si mañana cambia la estructura de los datos enviados por el frontend?

La validación de `PredictionRequest` detectaría datos faltantes, tipos incorrectos o valores no permitidos y la API respondería con un error controlado. Después habría que actualizar el frontend y, si cambian las variables del modelo, también entrenar y guardar una nueva versión del pipeline.
