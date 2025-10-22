import pandas as pd
from joblib import load

# Cargar pipeline y columnas usadas por el modelo
pipeline = load("model/api_pickle/pipeline_abandono.joblib")

with open("model/api_pickle/columnas_modelo3.txt", "r") as f:
    columnas_modelo = [line.strip() for line in f.readlines()]


# Cargar datos activos
df_validation = pd.read_csv('data/df_validacion_inicial.csv')

# Columnas booleanas esperadas en el modelo
bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos', 
    'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
    'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
    'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']