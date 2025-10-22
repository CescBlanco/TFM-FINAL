import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

def guardar_scaler(scaler, nombre_modelo):
    os.makedirs("tmp_artifacts", exist_ok=True)
    path = f"tmp_artifacts/scaler_{nombre_modelo}.pkl"
    joblib.dump(scaler, path)
    return path

def cargar_modelo_y_scaler(experimento, metric="auc"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experimento)
    if not experiment:
        raise ValueError(f"No se encontró el experimento {experimento} en MLflow")

    best_run = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )[0]
    run_id = best_run.info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Buscar y descargar scaler
    artifacts = client.list_artifacts(run_id)
    scaler_artifact = next(a.path for a in artifacts if "scaler" in a.path.lower())
    scaler_path = client.download_artifacts(run_id, scaler_artifact, "./tmp_artifacts")
    scaler = joblib.load(scaler_path)

    return model, scaler, run_id