import logging
from fastapi import FastAPI, HTTPException
from typing import Union

from model.api_pickle.schemas import UserData, MultiUserData, IDRequest, IDListRequest
from model.api_pickle.predictor import predecir_con_datos, predecir_por_id, predecir_por_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Abandono modular")

@app.get("/")
def index():
    """
    Endpoint raíz para comprobar que la API está viva.
    """
    return {"mensaje": "API de abandono flexible (datos, ID, múltiples IDs)"}

@app.post("/predecir/")
def predecir(data: Union[UserData, MultiUserData]):
    """
    Predicción de churn a partir de datos enviados (único o múltiples).
    """
    logger.info(f"Petición de predicción con datos: {data}")
    return predecir_con_datos(data)

@app.post("/predecir_por_id/")
def prediccion_id(request: IDRequest):
    """
    Predicción de churn a partir de un único IdPersona.
    """
    logger.info(f"Petición de predicción para IdPersona: {request.IdPersona}")
    return predecir_por_id(request.IdPersona)

@app.post("/predecir_por_ids/")
def prediccion_ids(request: IDListRequest):
    """
    Predicción de churn para una lista de IdPersona.
    """
    logger.info(f"Petición de predicción para múltiples Ids: {request.Ids}")
    return predecir_por_ids(request.Ids)