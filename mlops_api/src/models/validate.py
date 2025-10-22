import os
import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
import joblib
import shap
from src.config import VALIDATION_OUTPUT_PATH
from src.utils.artifacts import cargar_modelo_y_scaler
from src.utils.metrics import calcular_metricas
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def validar_modelo(experimento, archivo_validacion, metric="auc"):
    """
    Evalúa el mejor modelo de un experimento sobre un dataset de validación.
    """
    print(f"\n🧪 Validando experimento '{experimento}' usando {archivo_validacion}")

    model, scaler, run_id = cargar_modelo_y_scaler(experimento, metric)
    df_valid = pd.read_csv(os.path.join(VALIDATION_OUTPUT_PATH, archivo_validacion))

    X_valid = df_valid.drop(columns=["Abandono"], errors="ignore")
    y_valid = df_valid["Abandono"] if "Abandono" in df_valid.columns else None

    X_valid_scaled = scaler.transform(X_valid)
    y_pred = model.predict(X_valid_scaled)
    y_prob = model.predict_proba(X_valid_scaled)[:, 1]

    if y_valid is not None:
        metrics = calcular_metricas(y_valid, y_pred, y_prob)
        print("\n📊 Métricas de validación:\n", metrics)
        return metrics, y_pred, y_prob
    else:
        df_pred = df_valid.copy()
        df_pred["Prediccion"] = y_pred
        df_pred["Probabilidad"] = y_prob
        out_path = os.path.join(VALIDATION_OUTPUT_PATH, f"predicciones_{experimento}.csv")
        df_pred.to_csv(out_path, index=False)
        print(f"✅ Predicciones guardadas en: {out_path}")
        return df_pred
    

def obtener_caracteristicas_importantes(modelo, columnas, X_val):
    """
    Obtiene las características más importantes de un modelo basado en árboles o lineal.
    """

    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
    elif hasattr(modelo, 'coef_'):
        importances = abs(modelo.coef_[0])
    else:
        # Si no hay feature_importances_ o coef_, usamos SHAP
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_val)  # Aquí asumimos que tienes X_val
        importances = shap_values[0].mean(axis=0)  # Promedio de la importancia para cada característica
    
    # Empareja las importancias con las columnas y ordénalas
    return sorted(zip(columnas, importances), key=lambda x: x[1], reverse=True)


def evaluar_validacion_externa(experiment_name, features):
    """
    Valida el modelo de un experimento MLflow usando un dataset externo y registra resultados.
    Genera:
    - Predicciones por persona.
    - Importancias por persona.
    - Importancias globales del modelo.
    """

    val_path = f"{VALIDATION_OUTPUT_PATH}/df_validacion_{experiment_name}.csv"
    df_val = pd.read_csv(val_path)

    X_val = df_val[features]
    y_val = df_val['Abandono']
    ids_persona = df_val['IdPersona']  

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise Exception(f"❌ El experimento '{experiment_name}' no existe en MLflow")

    # Obtener el mejor run por AUC
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc DESC"]
    )[0]
    run_id = best_run.info.run_id

    # Cargar modelo
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Listar artefactos para encontrar el scaler
    artifacts= client.list_artifacts(run_id)
    scaler_artifact_name= None
    for artifact in artifacts:
        if "scaler" in artifact.path.lower():
            scaler_artifact_name= artifact.path
            break
    
    if scaler_artifact_name is None:
        raise Exception('No se encontró ningún archivo scaler en los artefactos del run.')
    
    #Descargar el scaler
    scaler_path = client.download_artifacts(run_id, scaler_artifact_name, "./tmp_artifacts")
    scaler = joblib.load(scaler_path)

    # Transformar los datos
    X_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # --- Métricas ---  
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_val = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    # Obtener las características más importantes
    caracteristicas_importantes = obtener_caracteristicas_importantes(model, features, X_val)
    df_importances_global = pd.DataFrame(caracteristicas_importantes, columns=['Feature','Importance'])
    global_importances_path = f"tmp_artifacts/importancias_global_{experiment_name}.csv"
    df_importances_global.to_csv(global_importances_path, index=False)
    
    # Importancias por persona
    feature_importances = model.feature_importances_ if hasattr(model, "feature_importances_") else model.coef_.flatten()
    importances_df = pd.DataFrame(X_scaled, columns=features)
    for i, f in enumerate(features):
        importances_df[f + '_importance'] = X_scaled[:, i] * feature_importances[i]
    importances_df['IdPersona'] = ids_persona
    person_importances_path = f"tmp_artifacts/importancias_persona_{experiment_name}.csv"
    importances_df.to_csv(person_importances_path, index=False)

    # Predicciones
    pred_df = pd.DataFrame({
        "IdPersona": ids_persona,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_prob": y_prob
    })
    pred_df['nivel_riesgo'] = pd.cut(pred_df['y_prob'], bins=[0,0.2,0.4,0.6,0.8,1],
                    labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])

    preds_path = f"tmp_artifacts/preds_{experiment_name}.csv"
    pred_df.to_csv(preds_path, index=False)

    # 🔐 Registrar modelo en el Model Registry
    MODEL_NAME = experiment_name  # Ej: "modelo3"
    try:
        result = mlflow.register_model( model_uri=f"runs:/{run_id}/model", name=MODEL_NAME )
    except MlflowException as e:
        if "already exists" in str(e):
            result = mlflow.register_model( model_uri=f"runs:/{run_id}/model", name=MODEL_NAME  )
        else:
            raise
    model_version = result.version

    # # Promover a Production y archivar versiones anteriores
    # client.transition_model_version_stage(
    #     name=MODEL_NAME,
    #     version=model_version,
    #     stage="Production",
    #     archive_existing_versions=True
    # )

    #print(f"✅ Modelo '{MODEL_NAME}' versión {model_version} registrado y movido a Production")

    # Logging de validación externa como nuevo run
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"validacion_externa_{experiment_name}"):
        mlflow.set_tags({"type": "validacion_externa", "validated_model": run_id })
        mlflow.log_param("modelo_validado", run_id)
        mlflow.log_metrics({
            "val_accuracy": float(acc),
            "val_recall": float(rec),
            "val_f1": float(f1),
            "val_auc": float(auc_val)
        })

        mlflow.log_param("caracteristicas_importantes", str(caracteristicas_importantes))
        mlflow.log_artifact(preds_path)
        mlflow.log_artifact(global_importances_path)
        mlflow.log_artifact(person_importances_path)


        # Limpiar tmp
        for f in [preds_path, global_importances_path, person_importances_path]:
            if os.path.exists(f):
                os.remove(f)


        

        # Print resumen
        print(f"\n🔎 Resultados validación externa ({experiment_name}):")
        print(f"Accuracy: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc_val:.3f}")
        print("Confusion Matrix:\n", cm)
        print("Top 5 características importantes:", caracteristicas_importantes[:5])
        print("✅ Archivos generados:")
        print("   - Predicciones por persona")
        print("   - Importancias globales")
        print("   - Importancias por persona")


