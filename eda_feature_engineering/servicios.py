from eda_feature_engineering.funciones import *

def run():
    df_servicios = load_dataset('data/Fisio-Nutri 01.09.2024 a 01.09.2025 FORMATO BUENO.xlsx', sheet_name="adaptado")

    columnas_a_eliminar = []
    columnas_a_renombrar = {}
    columnas_numericas = ['IdPersona','Importe_2024_servicios','Cantidad_2024_servicios', 'Importe_2025_servicios',
                    'Cantidad_2025_servicios', 'Importe_total_pagado_servicios', 'Cantidad_total_pagado_servicios']
    columnas_fechas= []

    df_servicios = preparar_datos_iniciales(df_servicios, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)
    
    eda_basica(df_servicios, nombre_df="Clientes con servicios extra")
    df_servicios=df_servicios.fillna(0)
    df_servicios.isnull().sum()

    df_servicios_agregados = agregar_servicios(df_servicios)
    df_servicios_encoded = aplicar_one_hot_encoding_servicios(df_servicios)
    df_servicios_final = agregar_y_unir_servicios(df_servicios_encoded, df_servicios_agregados)
    df_servicios_final = preparar_servicios_final(df_servicios_final)

    # Guardar el DataFrame en un archivo CSV
    df_servicios_final.to_csv('data/servicios_final.csv', index=False)

    return df_servicios_final