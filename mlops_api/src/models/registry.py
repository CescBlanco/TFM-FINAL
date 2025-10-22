from src.utils.artifacts import cargar_modelo_y_scaler

def obtener_modelo_produccion(nombre_experimento: str, metric: str = "auc") -> tuple:
    """
    Carga el modelo y el scaler del mejor run de un experimento MLflow para su uso en inferencia (como en una API).

    Esta función consulta MLflow para obtener el modelo entrenado con el mejor valor de una métrica de evaluación (por defecto "auc").
    El modelo y el scaler se cargan para ser utilizados en tareas de inferencia.

    Parámetros:
        nombre_experimento (str): Nombre del experimento MLflow para identificar el modelo a cargar.
        metric (str, opcional): Nombre de la métrica utilizada para identificar el mejor modelo (por defecto "auc").
        
    Retorna:
        tuple: Una tupla que contiene:
            - `model`: El modelo entrenado en el mejor run del experimento.
            - `scaler`: El scaler utilizado para transformar los datos de entrada en el mejor run del experimento.

    Excepciones:
        - MlflowException: Si el experimento no existe o no se encuentra el modelo o el scaler.
    """

    print(f"📦 Cargando modelo en producción para experimento '{nombre_experimento}'...")
    
    # Cargar el modelo y el scaler del mejor run de MLflow
    model, scaler, run_id = cargar_modelo_y_scaler(nombre_experimento, metric)
    print(f"✅ Modelo cargado (run_id: {run_id})")
    return model, scaler
