from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import mlflow.sklearn
import os
import joblib
import json
from datetime import datetime

# --- PATH para guardar predicciones ---
PREDICCIONES_PATH = "predicciones_api.csv"
mlflow.set_tracking_uri("file:///C:/Users/cescb/OneDrive/Documents/Proyecto_python_Evolve/TFM-FINAL/mlruns")

# --- Función utilitaria para cargar el scaler dinámicamente ---
def obtener_scaler_dinamico(client, run_id):
    artifacts = client.list_artifacts(run_id)
    scaler_path = None
    for artifact in artifacts:
        if 'scaler' in artifact.path.lower() and artifact.is_dir is False:
            scaler_path = artifact.path
            break
    if scaler_path is None:
        raise FileNotFoundError(f"No se encontró ningún scaler en los artefactos del run {run_id}")
    local_path = client.download_artifacts(run_id, scaler_path, "./tmp_artifacts")
    return joblib.load(local_path)

def validar_columnas_esperadas(df, columnas_esperadas):
    columnas_faltantes = set(columnas_esperadas) - set(df.columns)
    columnas_extra = set(df.columns) - set(columnas_esperadas)
    errores = []
    if columnas_faltantes:
        errores.append(f"Faltan columnas: {', '.join(columnas_faltantes)}")
    if columnas_extra:
        errores.append(f"Columnas inesperadas: {', '.join(columnas_extra)}")
    return errores

def obtener_importancia_por_persona(modelo, X_scaled, features):
    # Obtener las importancias globales (coeficientes o feature_importances_)
    feature_importances = modelo.feature_importances_ if hasattr(modelo, "feature_importances_") else modelo.coef_.flatten()
    
    # Crear un DataFrame de importancias por persona
    importances_df = pd.DataFrame(X_scaled, columns=features)
    
    # Calcular la importancia de cada característica para cada persona
    for i, f in enumerate(features):
        importances_df[f + '_importance'] = X_scaled[:, i] * feature_importances[i]
    
    return importances_df

def guardar_predicciones_api(idpersona, variables, pred, prob, nivel, endpoint, run_id, version_modelo=None):
    """
    Guarda la predicción con todas las columnas necesarias.
    """
    registro = {
        "IdPersona": idpersona,
        "VariablesEntrada": json.dumps(variables, ensure_ascii=False),
        "Prediccion": int(pred),
        "Probabilidad": float(prob),
        "NivelRiesgo": nivel,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Endpoint": endpoint,
        "RunIDModelo": run_id,
        "VersionModelo": version_modelo
    }

    # Si el CSV existe, anexamos, si no lo creamos
    if os.path.exists(PREDICCIONES_PATH):
        pd.DataFrame([registro]).to_csv(PREDICCIONES_PATH, mode='a', index=False, header=False)
    else:
        pd.DataFrame([registro]).to_csv(PREDICCIONES_PATH, index=False)

def obtener_mejor_modelo(experimento: str, metric_name: str = "AUC"):
    """
    Obtiene el mejor modelo registrado en MLFlow basado en una métrica específica (por defecto AUC).
    """
    client = MlflowClient()

    # Obtener el experimento
    try:
        experiment = client.get_experiment_by_name(experimento)
    except mlflow.exceptions.MlflowException:
        raise ValueError(f"El experimento {experimento} no existe en MLFlow.")
    
    # Obtener todos los runs asociados con ese experimento
    runs = client.search_runs(experiment.experiment_id, filter_string="", run_view_type=mlflow.tracking.client.ViewType.ALL)
    
    # Crear una lista de los resultados de las métricas de los modelos
    resultados = []
    for run in runs:
        run_id = run.info.run_id
        auc = run.data.metrics.get(metric_name, None)
        
        # Si el run tiene la métrica AUC, agregamos los resultados
        if auc is not None:
            resultados.append({
                "run_id": run_id,
                "auc": auc,
                "accuracy": run.data.metrics.get("accuracy", None),
                "f1_score": run.data.metrics.get("f1_score", None),
                "recall": run.data.metrics.get("recall", None)
            })
    
    # Convertir los resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Asegurarse de que haya resultados
    if df_resultados.empty:
        raise ValueError(f"No se encontraron métricas para el experimento {experimento}.")
    
    # Obtener el mejor modelo según AUC (o la métrica que prefieras)
    mejor_modelo_info = df_resultados.sort_values(by=metric_name, ascending=False).iloc[0]
    
    print(f"Mejor modelo encontrado: Run ID: {mejor_modelo_info['run_id']}, AUC: {mejor_modelo_info[metric_name]}")

    # Cargar el modelo usando el run_id
    model_uri = f"runs:/{mejor_modelo_info['run_id']}/model"
    modelo_final = mlflow.sklearn.load_model(model_uri)

    # Obtener el run_id del mejor modelo
    run_id = mejor_modelo_info["run_id"]
    
    # Obtener los artefactos del run (como el scaler)
    artifacts = client.list_artifacts(run_id)
    print("Artifacts en el run:")
    scaler_path = None
    for artifact in artifacts:
        if 'scaler' in artifact.path.lower() and artifact.is_dir is False:
            scaler_path = artifact.path
            break
    
    if scaler_path is None:
        raise FileNotFoundError(f"No se encontró un scaler en los artefactos del run {run_id}")
    
    # Descargar el scaler
    local_path = client.download_artifacts(run_id, scaler_path, "./tmp_artifacts")
    scaler = joblib.load(local_path)
    
    return modelo_final, mejor_modelo_info, scaler

# Aquí puedes definir el nombre del experimento y la métrica a usar
modelo, mejor_info, scaler = obtener_mejor_modelo("Experimento_v3", metric_name="auc")

print("Modelo cargado correctamente:", modelo)
print("Detalles del mejor modelo:", mejor_info)
print("Detalles del mejor scaler:", scaler)


# === Cargar columnas esperadas ===
with open("src/api/columnas_modelo3.txt") as f:
    columnas_modelo3 = f.read().splitlines()

# === Cargar dataset de validación (para endpoint por ID) ===
df_validation = pd.read_csv('data_mlops_api/df_validacion_inicial.csv')

# === Inicializar API ===
app = FastAPI(title="API Abandono flexible")

# ==== MODELOS DE DATOS ====

class UserData(BaseModel):
    Edad: int
    Sexo_Mujer: bool
    UsoServiciosExtra: bool
    ratio_cantidad_2025_2024: float
    Diversidad_servicios_extra: int
    TienePagos: bool
    TotalVisitas: int
    DiasActivo: int
    VisitasUlt90: int
    VisitasUlt180: int
    TieneAccesos: bool
    VisitasPrimerTrimestre: int
    VisitasUltimoTrimestre: int
    DiaFav_domingo: bool
    DiaFav_jueves: bool
    DiaFav_lunes: bool
    DiaFav_martes: bool
    DiaFav_miercoles: bool
    DiaFav_sabado: bool
    DiaFav_viernes: bool
    EstFav_invierno: bool
    EstFav_otono: bool
    EstFav_primavera: bool
    EstFav_verano: bool

        # Usando field_validator para Pydantic V2
    @field_validator('Edad')
    def edad_mayor_18(cls, value):
        if value < 18:
            raise ValueError('Edad debe ser mayor o igual a 18')
        return value
    
    @field_validator('ratio_cantidad_2025_2024')
    def ratio_no_negativo(cls, value):
        if value < 0:
            raise ValueError('El ratio no puede ser negativo')
        return value

    @field_validator('Diversidad_servicios_extra', 'TotalVisitas', 'DiasActivo', 'VisitasUlt90',
                      'VisitasUlt180', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre')
    def valores_no_negativos(cls, value, field):
        if value < 0:
            raise ValueError(f"{field.name} no puede ser negativo")
        return value

class MultiUserData(BaseModel):
    datos: List[UserData]


class IDRequest(BaseModel):
    IdPersona: int


class IDListRequest(BaseModel):
    Ids: List[int]


@app.get("/", summary='Mensaje de bienvenida')
def index():
    return {"mensaje": "API de predicción de abandono con modelo entrenado en MLflow"}


@app.post("/predecir_abandono/", summary="Predicción por datos completos (uno o varios)")
def predecir_abandono(data: Union[UserData, MultiUserData]):
    if isinstance(data, UserData):
        df = pd.DataFrame([data.dict()])
    else:
        df = pd.DataFrame([d.dict() for d in data.datos])

    # Asegurarse de que las columnas booleanas se conviertan a enteros
    bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                 'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                 'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                 'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)
    df = df[columnas_modelo3]
    errores = validar_columnas_esperadas(df, columnas_modelo3)
    if errores:
        raise HTTPException(status_code=400, detail="; ".join(errores))

    X_scaled = scaler.transform(df)

    prediccion = modelo.predict(X_scaled)[0]
    probabilidad = modelo.predict_proba(X_scaled)[0][1]  # Probabilidad de clase 1 (abandono)

        # Categorizar el nivel de riesgo
    niveles = pd.cut(
        probabilidad,
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
        include_lowest=True
    )
    # Obtener la importancia de las características por persona
    importancia_por_persona_df = obtener_importancia_por_persona(modelo, X_scaled, df.columns)
    # Crear los resultados
    resultados = [{
        
        "ProbabilidadAbandono": round(probabilidad, 3),
        "NivelRiesgo": niveles,
        "CaracterísticasImportantes": importancia_por_persona_df.to_dict(orient="records")  # Devolvemos los resultados de las importancias
        
    }]
    
    return resultados[0] if isinstance(data, UserData) else resultados

@app.post("/predecir_abandono_por_id/", summary='Predicción por IdPersona')
def predecir_abandono_por_id(request: IDRequest):
    id_buscar = request.IdPersona
    fila = df_validation[df_validation['IdPersona'] == id_buscar]
    
    if fila.empty:
        raise HTTPException(status_code=404, detail=f"IdPersona {id_buscar} no encontrado")

    df = fila[columnas_modelo3].copy()
    # Asegurarse de que las columnas booleanas se conviertan a enteros
    bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                 'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                 'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                 'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)

    errores = validar_columnas_esperadas(df, columnas_modelo3)
    if errores:
        raise HTTPException(status_code=400, detail="; ".join(errores))
    X_scaled = scaler.transform(df)

    prediccion = modelo.predict(X_scaled)[0]
    probabilidad = modelo.predict_proba(X_scaled)[0][1]  #

    
    # Categorizar el nivel de riesgo
    nivel = pd.cut([probabilidad], bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                   labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
                   include_lowest=True)[0]

    importancia_por_persona_df = obtener_importancia_por_persona(modelo, X_scaled, df.columns)
    return {
        "IdPersona": int(id_buscar),
        "ProbabilidadAbandono": round(probabilidad, 3),
        "NivelRiesgo": nivel,
        "CaracterísticasImportantes": importancia_por_persona_df.to_dict(orient="records")  # Devolvemos los resultados de las importancias
    }

@app.post("/predecir_abandono_por_ids/", summary='Predicción por lista de IdPersona')
def predecir_abandono_por_ids(request: IDListRequest):
    resultados = []

    for idpersona in request.Ids:
        fila = df_validation[df_validation['IdPersona'] == idpersona]
        if fila.empty:
            resultados.append({
                "IdPersona": idpersona,
                "error": "No encontrado"
            })
            continue

        df = fila[columnas_modelo3].copy()
        # Asegurarse de que las columnas booleanas se conviertan a enteros
        bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                     'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                     'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                     'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
        df[bool_cols] = df[bool_cols].astype(int)

        errores = validar_columnas_esperadas(df, columnas_modelo3)
        if errores:
            raise HTTPException(status_code=400, detail="; ".join(errores))
        X_scaled = scaler.transform(df)

        prediccion = modelo.predict(X_scaled)[0]
        probabilidad = modelo.predict_proba(X_scaled)[0][1]  #

        
        # Categorizar el nivel de riesgo
        nivel = pd.cut([probabilidad], bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                       labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
                       include_lowest=True)[0]
        # Obtener las características más importantes
        importancia_por_persona_df = obtener_importancia_por_persona(modelo, X_scaled, df.columns)

        resultados.append({
            "IdPersona": idpersona,
            "ProbabilidadAbandono": round(probabilidad, 3),
            "NivelRiesgo": nivel,
            "CaracterísticasImportantes": importancia_por_persona_df.to_dict(orient="records")  # Devolvemos los resultados de las importancias
        })

    return resultados
