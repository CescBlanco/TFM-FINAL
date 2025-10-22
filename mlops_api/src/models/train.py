import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn




from src.load_data.loader import cargar_datos
from src.utils.plotting import plot_confusion_matrix
from src.utils.metrics import calcular_metricas
from src.utils.artifacts import *

def entrenar_modelos(nombre_experimento, features):
    X, y = cargar_datos(nombre_experimento, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "SVM": {"C": [0.1, 1, 10]},
        "KNN": {"n_neighbors": [3, 5, 7]}
    }

    mlflow.set_experiment(nombre_experimento)
    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Entrenando: {nombre}")
        start = time.time()

        grid = GridSearchCV(modelo, param_grids[nombre], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        metrics = calcular_metricas(y_test, y_pred, y_prob)

        with mlflow.start_run(run_name=nombre):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "type": "training",
                "model_name": nombre,
                "feature_set": nombre_experimento
            })

            mlflow.sklearn.log_model(best_model, "model")

            scaler_path = guardar_scaler(scaler, nombre)
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)

            plot_path = f"tmp_artifacts/cm_{nombre}.png"
            plot_confusion_matrix(y_test, y_pred, plot_path)
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, f"{nombre_experimento}_{nombre.replace(' ', '_')}")

        resultados.append({
            "Modelo": nombre,
            **metrics,
            "Tiempo": round(time.time() - start, 2)
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by="auc", ascending=False)
    print("\n🏁 Resultados finales:\n", df_resultados)
    return df_resultados