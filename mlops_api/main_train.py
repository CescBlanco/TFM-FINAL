from src.models.train import entrenar_modelos
from src.config import *

import mlflow

mlflow.set_tracking_uri("file:///C:/Users/cescb/OneDrive/Documents/Proyecto_python_Evolve/TFM-FINAL/mlruns")


entrenar_modelos("Experimento_v1", FEATURES_1)
entrenar_modelos("Experimento_v2", FEATURES_2)
entrenar_modelos("Experimento_v3", FEATURES_3)