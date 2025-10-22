from src.models.validate import *
from src.config import *

import mlflow

mlflow.set_tracking_uri("file:///C:/Users/cescb/OneDrive/Documents/Proyecto_python_Evolve/TFM-FINAL/mlruns")


evaluar_validacion_externa("Experimento_v1", FEATURES_1)
evaluar_validacion_externa("Experimento_v2", FEATURES_2)
evaluar_validacion_externa("Experimento_v3", FEATURES_3)