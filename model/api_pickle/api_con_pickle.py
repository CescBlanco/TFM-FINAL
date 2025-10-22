from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from joblib import load
import pandas as pd

# Carga pipeline, columnas y datos activos
pipeline = load("model/api_pickle/pipeline_abandono.joblib")

with open("model/api_pickle/columnas_modelo3.txt") as f:
    columnas_modelo3= f.read().splitlines()

df_validation = pd.read_csv('data/df_validacion_inicial.csv')  # Tiene IdPersona + columnas_modelo

app = FastAPI(title="API Abandono flexible")

# Modelo para input completo (una fila)
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


# Modelo para input múltiple (lista de filas)
class MultiUserData(BaseModel):
    datos: List[UserData]

# Modelo para solicitar por Id
class IDRequest(BaseModel):
    IdPersona: int

@app.get("/")
def index():
    return {"mensaje": "API de churn flexible con predicción por datos o por ID"}

# Endpoint para predecir pasando lista o una sola fila completa (datos)
@app.post("/predecir/")
def predecir_churn(data: Union[UserData, MultiUserData]):
    # Crear DataFrame desde input, ya sea uno o varios
    if isinstance(data, UserData):
        df = pd.DataFrame([data.dict()])
    else:
        df = pd.DataFrame([d.dict() for d in data.datos])
    
    bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos', 
                'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)
    df = df[columnas_modelo3]

    probs = pipeline.predict_proba(df)[:, 1]

    niveles = pd.cut(
        probs,
        bins=[0,0.2,0.4,0.6,0.8,1],
        labels=["Muy Bajo","Bajo","Medio","Alto","Muy Alto"],
        include_lowest=True
    )

    resultados = []
    for prob, nivel in zip(probs, niveles):
        resultados.append({
            "ProbabilidadAbandono": round(prob, 3),
            "NivelRiesgo": nivel
        })

    # Si era input único, devolver dict único, sino lista
    return resultados[0] if isinstance(data, UserData) else resultados

# Endpoint para predecir solo pasando IdPersona
@app.post("/predecir_por_id/")
def predecir_por_id(request: IDRequest):
    id_buscar = request.IdPersona
    fila = df_validation[df_validation['IdPersona'] == id_buscar]
    if fila.empty:
        raise HTTPException(status_code=404, detail=f"IdPersona {id_buscar} no encontrado")

    df = fila[columnas_modelo3].copy()
    bool_cols = ['Sexo_Mujer','UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)

    prob = pipeline.predict_proba(df)[0][1]
    nivel = pd.cut([prob], bins=[0,0.2,0.4,0.6,0.8,1], labels=["Muy Bajo","Bajo","Medio","Alto","Muy Alto"], include_lowest=True)[0]

    return {
        "IdPersona": int(id_buscar),
        "ProbabilidadAbandono": round(prob, 3),
        "NivelRiesgo": nivel
    }


# Nuevo modelo para lista de IDs
class IDListRequest(BaseModel):
    Ids: List[int]

@app.post("/predecir_por_ids/")
def predecir_por_ids(request: IDListRequest):
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
        bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                    'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                    'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                    'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
        df[bool_cols] = df[bool_cols].astype(int)

        prob = pipeline.predict_proba(df)[0][1]
        nivel = pd.cut([prob], bins=[0,0.2,0.4,0.6,0.8,1],
                       labels=["Muy Bajo","Bajo","Medio","Alto","Muy Alto"], include_lowest=True)[0]

        resultados.append({
            "IdPersona": idpersona,
            "ProbabilidadAbandono": round(prob, 3),
            "NivelRiesgo": nivel
        })

    return resultados
