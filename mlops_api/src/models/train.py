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

def entrenar_modelos(nombre_experimento: str, features: list) -> pd.DataFrame:
    """
    Entrena varios modelos de clasificación utilizando GridSearchCV, evalúa su rendimiento, y registra los resultados
    en MLflow. Los modelos entrenados incluyen: Regresión Logística, Random Forest, Gradient Boosting, SVM y KNN.

    Parámetros:
        nombre_experimento (str): Nombre del experimento que se utiliza para configurar el entorno de MLflow y 
                                   guardar los artefactos y métricas relacionados.
        features (list): Lista de nombres de las columnas que se utilizarán como características para el entrenamiento 
                         del modelo.

    Retorna:
        pd.DataFrame: Un DataFrame que contiene los resultados de las métricas de evaluación (como AUC, accuracy, etc.)
                      para cada modelo entrenado, ordenado por la métrica AUC de mayor a menor.

    Guardado:
        - Los modelos entrenados se registran en MLflow junto con sus métricas y parámetros.
        - Se guarda el scaler utilizado para normalizar los datos.
        - Se guarda una imagen de la matriz de confusión de cada modelo.
        
    Excepciones:
        FileNotFoundError: Si no se encuentra el archivo de datos necesario en el `DATA_PATH`.
        KeyError: Si alguna de las columnas esperadas no está presente en el DataFrame cargado.
    """
    
    # Cargar los datos desde la funcion definida en el archivo loader de la carpeta load_data.
    X, y = cargar_datos(nombre_experimento, features)

    # Dividir los datos en entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Inicializar el escalador y aplicar la normalización.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir los modelos a entrenar.
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
        }

    # Definir las grillas de parámetros para cada modelo.
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "SVM": {"C": [0.1, 1, 10]},
        "KNN": {"n_neighbors": [3, 5, 7]}
        }

    # Configurar el experimento en MLflow.
    mlflow.set_experiment(nombre_experimento)
    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Entrenando: {nombre}")
        start = time.time()
        
        # Realizar la búsqueda en cuadrícula con validación cruzada.
        grid = GridSearchCV(modelo, param_grids[nombre], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_

        # Hacer predicciones.
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        # Calcular métricas de evaluación.
        metrics = calcular_metricas(y_test, y_pred, y_prob)
        
        # Registrar el modelo y las métricas en MLflow.
        with mlflow.start_run(run_name=nombre):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "type": "training",
                "model_name": nombre,
                "feature_set": nombre_experimento
            })

            mlflow.sklearn.log_model(best_model, "model")

             # Guardar el scaler como artefacto.
            scaler_path = guardar_scaler(scaler, nombre)
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)

            # Guardar la matriz de confusión.
            plot_path = f"tmp_artifacts/cm_{nombre}.png"
            plot_confusion_matrix(y_test, y_pred, plot_path)
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            # Registrar el modelo en el catálogo de modelos de MLflow.
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, f"{nombre_experimento}_{nombre.replace(' ', '_')}")

        # Guardar los resultados para cada modelo.
        resultados.append({
            "Modelo": nombre,
            **metrics,
            "Tiempo": round(time.time() - start, 2)
        })

    # Ordenar los resultados por la métrica AUC.
    df_resultados = pd.DataFrame(resultados).sort_values(by="auc", ascending=False)
    print("\n🏁 Resultados finales:\n", df_resultados)
    
    return df_resultados