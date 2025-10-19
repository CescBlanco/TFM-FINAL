from eda_feature_engineering.funciones import *

def run():
    
    abonados = load_dataset('data/abonados_final_pre_modelo.csv')
    servicios_extra = load_dataset('data/servicios_final.csv')
    economia = load_dataset('data/economia_final_pre_modelo.csv')
    accesos = load_dataset('data/resumen_accesos_pre_modelo.csv')

    df_abonados_servicios = merge_y_limpiar_usuarios_servicios(abonados, servicios_extra)

    eda_basica(df_abonados_servicios, nombre_df="Clientes con Servicios Extra")

    cols_a_borrar = [
        'Importe_2024_servicios', 'Cantidad_2024_servicios', 'Importe_2025_servicios', 'Cantidad_2025_servicios',
        'Concepto_ENTRENADOR PERSONAL 1 /2 SESSIO ABONAT', 'Concepto_ENTRENADOR PERSONAL 1 SESSIO',
        'Concepto_ENTRENADOR PERSONAL 1 SESSIÓ ABONAT', 'Concepto_ENTRENADOR PERSONAL 10 (30 MINUTS)',
        'Concepto_ENTRENADOR PERSONAL 10 (30 MINUTS) ABONAT', 'Concepto_ENTRENADOR PERSONAL 10 SESSIONS ABONAT',
        'Concepto_ENTRENADOR PERSONAL 5 (30 MINUTS) ABONAT', 'Concepto_ENTRENADOR PERSONAL 5 SESSIONS ABONAT',
        "Concepto_FISIO ABONAMENT 10 SESSIONS 60' ABONAT", "Concepto_FISIO ABONAMENT 5 SESSIONS  30' ABONAT",
        "Concepto_FISIO ABONAMENT 5 SESSIONS  60'  ", 'Concepto_FISIO PACK BENVINGUDA',
        "Concepto_FISIOTERÀPIA  60' ABONAT", "Concepto_FISIOTERÀPIA 30' ABONAT", 'Concepto_NUTRI ANTROPOMETRIA',
        'Concepto_NUTRI ANTROPOMETRIA ABONAT', 'Concepto_NUTRI PACK 3 SEGUIMENTS ABONAT',
        'Concepto_NUTRI VISITA DE SEGUIMENT ABONAT', 'Concepto_PACK 3 SEGUIMENTS ABONAT',
        'Concepto_PACK NUTRICIÓ (2 VISITES) ABONAT', 'Total_tipos_servicios_unicos', 'Total_conceptos_unicos'
    ]

    df_final_features = crear_features_servicios(df_abonados_servicios, columnas_a_eliminar=cols_a_borrar)

    df = df_final_features[df_final_features['Edad'] != 225]

    union_economia = unir_con_features_economia(df, economia)
    union_economia_preparado = preparar_df_union_economia(union_economia)

    df_total_accesos= unir_df_con_features_accesos(union_economia_preparado, accesos)

    union_accesos_preparado = preparar_final_accesos(df_total_accesos)

    eda_basica(union_accesos_preparado, nombre_df="Dataframe final abonados")

    # Guardar el DataFrame en un archivo CSV
    union_accesos_preparado.to_csv('data/dataframe_final_abonado.csv', index=False)


    return union_accesos_preparado