from model.funciones_modelo import *
from datetime import datetime
import os


def run():

# ------------------------------------------------------------------
            # PREPARACIÓN DEL MODELO
# ------------------------------------------------------------------

    # Crear carpeta de imágenes con timestamp único por ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_base = f"model/imagenes/ejecucion_{timestamp}"
    carpeta_modelo_1 = os.path.join(carpeta_base, "modelo_1")
    carpeta_modelo_2 = os.path.join(carpeta_base, "modelo_2")
    carpeta_modelo_3 = os.path.join(carpeta_base, "modelo_3")

    for carpeta in [carpeta_modelo_1, carpeta_modelo_2, carpeta_modelo_3]:
        os.makedirs(carpeta, exist_ok=True)



    # Carga el dataset desde archivo CSV
    df = load_dataset('data/dataframe_final_abonado.csv')

    # Filtra solo los clientes mayores o iguales a 18 años
    df_clientes_adultos= df[df['Edad']>=18].reset_index(drop=True)
    
    # Renombra la columna objetivo 'EsChurn' a 'Abandono' para estandarización
    df = df_clientes_adultos.rename(columns= {'EsChurn': 'Abandono', 'DiaFav_miércoles': 'DiaFav_miercoles',
       'DiaFav_sábado': 'DiaFav_sabado'})

    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable
    eda_basica(df, nombre_df="Dataframe final abonados")

    # Análisis visual y estadístico de la variable target ('Abandono')
    analizar_target_abandono(df, target_col='Abandono')

    # Copia del dataset para el modelado
    df_modelo= df.copy()
    
    # Separa el 20% balanceado del dataset (10% de cada clase) como conjunto de validación final
    df_validacion, df_train= separacion_df_inferencia_test_final(df_modelo)

    # Muestra el conjunto de validación para inspección
    print(df_validacion)

    # Guardar el DataFrame en un archivo CSV.
    df_validacion.to_csv('data/df_validacion_inicial.csv', index=False)
    print("\n Dataframe inicial para la validación guardado en la carpeta data!\n")

    print("\n🚀 Inicio del entrenamiento de modelos...\n")

    print("\n-----------------MODELO 1-----------------------\n")
    # ------------------------------------------------------------------
                # MODELO 1
    # ------------------------------------------------------------------

    # Define las variables predictoras para el modelo
    X_1=  df_train[['Edad', 'VidaGymMeses', 'Sexo_Mujer', 
        'UsoServiciosExtra','ratio_cantidad_2025_2024',
        'Diversidad_servicios_extra','TotalPagadoEconomia',
        'TotalVisitas','DiasActivo','VisitasUlt90', 'VisitasUlt180',
        'TieneAccesos', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre', 'DiaFav_domingo',
       'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes', 'DiaFav_miercoles',
       'DiaFav_sabado', 'DiaFav_viernes', 'EstFav_invierno', 'EstFav_otono',
       'EstFav_primavera', 'EstFav_verano', 
        ]]

    # Variable objetivo
    y_1 = df_train['Abandono']

    # Aplica entrenamiento, optimización, evaluación y selección del mejor modelo
    best_model_1, scaler_1 = aplicacion_modelo(X_1, y_1, carpeta_modelo_1)

    # Crea una copia del dataset de validación para aplicar el modelo entrenado
    df_validacion_modelo1=  df_validacion.copy()

    # Realiza la predicción sobre el conjunto de validación (probabilidades)
    y_val_prob_1 = validacion_inferencia_testfinal(X_1, scaler_1, best_model_1, df_validacion_modelo1, carpeta_modelo_1)

    # Clasifica el riesgo de abandono en categorías y añade al dataset final
    df_validacion_modelo1= clasificar_riesgo(df_validacion_modelo1, y_val_prob_1)

    print("\n-----------------MODELO 2-----------------------\n")
  # ------------------------------------------------------------------
                # MODELO 2
    # ------------------------------------------------------------------

    # Define las variables predictoras para el modelo
    X_2=  df_train[['Edad', 'VidaGymMeses', 'Sexo_Mujer', 
        'UsoServiciosExtra','ratio_cantidad_2025_2024',
        'Diversidad_servicios_extra',#'TotalPagadoEconomia',
        'TotalVisitas','DiasActivo','VisitasUlt90', 'VisitasUlt180',
        'TieneAccesos', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre', 'DiaFav_domingo',
       'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes', 'DiaFav_miercoles',
       'DiaFav_sabado', 'DiaFav_viernes', 'EstFav_invierno', 'EstFav_otono',
       'EstFav_primavera', 'EstFav_verano', 
        ]]

    # Variable objetivo
    y_2 = df_train['Abandono']

    # Aplica entrenamiento, optimización, evaluación y selección del mejor modelo
    best_model_2, scaler_2 = aplicacion_modelo(X_2, y_2, carpeta_modelo_2)

    # Crea una copia del dataset de validación para aplicar el modelo entrenado
    df_validacion_modelo2=  df_validacion.copy()

    # Realiza la predicción sobre el conjunto de validación (probabilidades)
    y_val_prob_2 = validacion_inferencia_testfinal(X_2, scaler_2, best_model_2, df_validacion_modelo2, carpeta_modelo_2)

    # Clasifica el riesgo de abandono en categorías y añade al dataset final
    df_validacion_modelo2= clasificar_riesgo(df_validacion_modelo2, y_val_prob_2)

    print("\n-----------------MODELO 3-----------------------\n")

    # ------------------------------------------------------------------
                # MODELO 3
    # ------------------------------------------------------------------

    # Define las variables predictoras para el modelo
    X_3 = df_train[[
        'Edad','Sexo_Mujer', 'UsoServiciosExtra',
        'ratio_cantidad_2025_2024', 'Diversidad_servicios_extra',
        'TotalVisitas', 'DiasActivo', 'VisitasUlt90', 'VisitasUlt180',
        'TieneAccesos', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre',
        'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
        'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
        'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano'
        ]]


    # Variable objetivo
    y_3 = df_train['Abandono']

    # Aplica entrenamiento, optimización, evaluación y selección del mejor modelo
    best_model_3, scaler_3 = aplicacion_modelo(X_3, y_3, carpeta_modelo_3)

    # Crea una copia del dataset de validación para aplicar el modelo entrenado
    df_validacion_modelo3=  df_validacion.copy()

    # Realiza la predicción sobre el conjunto de validación (probabilidades)
    y_val_prob_3 = validacion_inferencia_testfinal(X_3, scaler_3, best_model_3, df_validacion_modelo3, carpeta_modelo_3)

    # Clasifica el riesgo de abandono en categorías y añade al dataset final
    df_validacion_modelo3= clasificar_riesgo(df_validacion_modelo3, y_val_prob_3)

    # Devuelve los datasets de validaciones y el contenido de los diferentes modelos con columnas de probabilidad y nivel de riesgo
    return (df_validacion_modelo1, df_validacion_modelo2, df_validacion_modelo3), (best_model_1, scaler_1, best_model_2, scaler_2,  best_model_3, scaler_3, X_1,X_2, X_3)
