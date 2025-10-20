from model.funciones_modelo import *

def run():

    # Carga el dataset desde archivo CSV
    df = load_dataset('data/dataframe_final_abonado.csv')

    # Filtra solo los clientes mayores o iguales a 18 años
    df_clientes_adultos= df[df['Edad']>=18].reset_index(drop=True)
    
    # Renombra la columna objetivo 'EsChurn' a 'Abandono' para estandarización
    df = df_clientes_adultos.rename(columns= {'EsChurn': 'Abandono'})

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

    # Define las variables predictoras para el modelo
    X_3 = df_train[[
        'Edad','Sexo_Mujer', 'UsoServiciosExtra',
        'ratio_cantidad_2025_2024', 'Diversidad_servicios_extra',
        'TotalVisitas', 'DiasActivo', 'VisitasUlt90', 'VisitasUlt180',
        'TieneAccesos', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre',
        'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
        'DiaFav_miércoles', 'DiaFav_sábado', 'DiaFav_viernes',
        'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano'
        ]]

    # Variable objetivo
    y_3 = df_train['Abandono']

    # Aplica entrenamiento, optimización, evaluación y selección del mejor modelo
    best_model, scaler= aplicacion_modelo(X_3, y_3)

    # Crea una copia del dataset de validación para aplicar el modelo entrenado
    df_validacion_modelo3=  df_validacion.copy()

    # Realiza la predicción sobre el conjunto de validación (probabilidades)
    y_val_prob_3= validacion_inferencia_testfinal(X_3, scaler, best_model, df_validacion_modelo3)

    # Clasifica el riesgo de abandono en categorías y añade al dataset final
    df_validacion_modelo3= clasificar_riesgo(df_validacion_modelo3, y_val_prob_3)

    # Devuelve el dataset de validación con columnas de probabilidad y nivel de riesgo
    return df_validacion_modelo3