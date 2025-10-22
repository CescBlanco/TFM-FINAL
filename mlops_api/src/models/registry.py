from src.utils.artifacts import cargar_modelo_y_scaler

def obtener_modelo_produccion(nombre_experimento, metric="auc"):
    """
    Retorna el modelo y scaler del mejor run de MLflow para usar en inferencia (API).
    """
    print(f"📦 Cargando modelo en producción para experimento '{nombre_experimento}'...")
    model, scaler, run_id = cargar_modelo_y_scaler(nombre_experimento, metric)
    print(f"✅ Modelo cargado (run_id: {run_id})")
    return model, scaler
