import os

DATA_PATH = os.path.join( "data_mlops_api", "dataframe_final_abonado.csv")
VALIDATION_OUTPUT_PATH = os.path.join( "data_mlops_api")


FEATURES_1 = [
    'Edad', 'VidaGymMeses', 'Sexo_Mujer', 'UsoServiciosExtra',
    'ratio_cantidad_2025_2024', 'Diversidad_servicios_extra',
    'TotalPagadoEconomia', 'TotalVisitas','TienePagos' ,'DiasActivo',
    'VisitasUlt90', 'VisitasUlt180', 'TieneAccesos', 
    'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre',
    'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
    'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
    'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano'
]

FEATURES_2 = [f for f in FEATURES_1 if f != "TotalPagadoEconomia"]
FEATURES_3 = [f for f in FEATURES_2 if f != "VidaGymMeses"]
