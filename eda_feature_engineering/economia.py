from eda_feature_engineering.funciones import *

def run():
    economia = load_dataset('data/Economia per persona 01.09.2024 a 01.09.2025.xlsx')

    columnas_a_eliminar = ['IdRemesa', 'FormaPagoMetálicoCantidad','FormaPagoMetálicoImporte','FormaPagoRecibo_domiciliadoCantidad', 
        'FormaPagoRecibo_domiciliadoImporte', 'FormaPagoTarjeta_créditoCantidad','FormaPagoTarjeta_créditoImporte',
         'FormaPago_Transf_HortaEsportivaCantidad','FormaPago_Transf_HortaEsportivaImporte', 'TotalCantidad', 'TotalImporte']

    columnas_a_renombrar = {'IdUsuario': 'IdPersona','FormaPagoMetálicoImporteCobrado': 'PagoMetálico','FormaPagoRecibo_domiciliadoImporteCobrado': 'PagoRecibo',
        'FormaPagoTarjeta_créditoImporteCobrado': 'PagoTarjeta','FormaPago_Transf_HortaEsportivaImporteCobrado': 'PagoTransferencia',  'TotalImporteCobrado': 'TotalCobrado'}

    columnas_numericas = ['PagoMetálico',
        'PagoRecibo', 'PagoTarjeta', 'PagoTransferencia', 'TotalCobrado']
    columnas_fechas = ['FechaRenovacion']

    economia_eda = preparar_datos_iniciales(economia, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)
    economia_eda['IdPersona'] = economia_eda['IdPersona'].astype(str)

    eda_basica(economia_eda, nombre_df="Clientes Economia")

    economia_eda= economia_eda.fillna(0)
    economia_eda.isnull().sum()

    
    economia_eda_filtrado= excluir_valores(economia_eda, 'TipoAbono', TIPOS_ABONO_EXCLUIR)
    df_features = economia_eda_filtrado.copy()
    df_features = df_features.sort_values(['IdPersona', 'FechaRenovacion'])

    df_features_economia = df_features.groupby('IdPersona').apply(agregar_features_economia).reset_index()
    df_features_economia['TienePagos'] = True
    df_features_economia= df_features_economia.fillna(0)

    # Guardar el DataFrame en un archivo CSV
    df_features_economia.to_csv('data/economia_final_pre_modelo.csv', index=False)
    
    return df_features_economia

