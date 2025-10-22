import pandas as pd
from typing import Union, List
from fastapi import HTTPException
import logging

from model.api_pickle.schemas import UserData, MultiUserData
from model.api_pickle.data import df_validation, pipeline, columnas_modelo, bool_cols


logger = logging.getLogger(__name__)

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el dataframe para predicción:
    - Convierte columnas booleanas a enteros
    - Asegura que todas las columnas del modelo estén presentes
    """
    df = df.copy()

    # Asegurar que todas las columnas del modelo estén presentes
    for col in columnas_modelo:
        if col not in df.columns:
            logger.warning(f"Columna {col} ausente en dataframe, agregando con valor 0")
            df[col] = 0

    # Convertir booleanos a enteros solo en las columnas que existan
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df[columnas_modelo]

def categorizar(prob: float) -> str:
    """
    Categoriza la probabilidad en niveles de riesgo.
    """
    return pd.cut(
        [prob],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
        include_lowest=True
    )[0]

def predecir_con_datos(data: Union[UserData, MultiUserData]):
    """
    Realiza predicción a partir de datos ingresados directamente.
    Puede recibir un único registro o una lista de registros.
    """
    if isinstance(data, UserData):
        df = pd.DataFrame([data.dict()])
    else:
        df = pd.DataFrame([d.dict() for d in data.datos])

    df = limpiar_dataframe(df)
    probs = pipeline.predict_proba(df)[:, 1]

    resultados = []
    for prob in probs:
        resultados.append({
            "ProbabilidadAbandono": round(prob, 3),
            "NivelRiesgo": categorizar(prob)
        })

    return resultados[0] if isinstance(data, UserData) else resultados

def predecir_por_id(idpersona: int):
    fila = df_validation[df_validation['IdPersona'] == idpersona]
    if fila.empty:
        logger.warning(f"IdPersona {idpersona} no encontrado en df_activos.")
        raise HTTPException(status_code=404, detail=f"IdPersona {idpersona} no encontrado")

    df_limpio = limpiar_dataframe(fila)  # Cambiar nombre variable local para evitar conflicto
    prob = pipeline.predict_proba(df_limpio)[0][1]

    return {
        "IdPersona": idpersona,
        "ProbabilidadAbandono": round(prob, 3),
        "NivelRiesgo": categorizar(prob)
    }

def predecir_por_ids(ids: List[int]):
    """
    Realiza predicción para múltiples IdPersona.
    Devuelve lista de resultados, con error si algún ID no es encontrado.
    """
    resultados = []

    for idpersona in ids:
        try:
            resultado = predecir_por_id(idpersona)
        except HTTPException:
            resultado = {
                "IdPersona": idpersona,
                "error": "No encontrado"
            }
            logger.info(f"ID {idpersona} no encontrado, agregado a resultados con error.")
        resultados.append(resultado)

    return resultados
