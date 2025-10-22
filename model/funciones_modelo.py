#IMPORTACIÓN LIBRERIAS NECESARIAS
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
from datetime import datetime

from IPython.display import display
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_curve, auc
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# FUNCIÓN PARA LA CARGA DE ARCHIVOS
def load_dataset(file_path, file_type=None, separator=None, encoding='utf-8', **kwargs):
    """
    Loads a dataset in different formats, with support for custom separators, encoding, and more options.
    """
    # If the file type is not specified, infer from file extension
    if not file_type:
        file_type = file_path.split('.')[-1].lower()

    # Load according to the file type
    if file_type == 'csv':
        return pd.read_csv(file_path, sep=separator or ',', encoding=encoding, **kwargs)
    elif file_type in ['xls', 'xlsx']:
        return pd.read_excel(file_path, **kwargs)
    elif file_type == 'json':
        return pd.read_json(file_path, encoding=encoding, **kwargs)
    else:
        raise ValueError(f"File format '{file_type}' not supported. Use 'csv', 'excel', or 'json'.")


#FUNCION PARA EL EDA BÁSICO
def eda_basica(df: pd.DataFrame, nombre_df: str = "DataFrame") -> None:
    """
    Realiza un análisis exploratorio básico sobre un DataFrame:
    - Identifica variables numéricas y categóricas
    - Detecta valores nulos y muestra una visualización si los hay
    - Revisa duplicados (filas y columnas)

    Parámetros:
        df (pd.DataFrame): El DataFrame a analizar
        nombre_df (str): Nombre para mostrar del DataFrame (opcional)
    """
    print(f"\n📋 Análisis EDA básico de: {nombre_df}")

    # 1. Tipos de variables
    print("\n📌 Tipos de Variables:")
    num_vbles = df.select_dtypes(include='number').columns.tolist()
    cat_vbles = df.select_dtypes(exclude='number').columns.tolist()
    print(f"🔢 Variables Numéricas: {num_vbles}")
    print(f"🔠 Variables Categóricas: {cat_vbles}")

    # 2. Valores nulos
    print("\n🕳️ Variables con valores nulos:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_percentage = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Total Missing': missing,
        'Percentage Missing': missing_percentage
    })

    if not missing.empty:
        display(missing_df)
        plt.figure(figsize=(10, 6))
        missing.plot(kind='barh', color='salmon')
        plt.title("Variables con Valores Nulos")
        plt.xlabel("Cantidad de valores nulos")
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("✅ No hay valores nulos en el dataset.")

    # 3. Filas duplicadas
    print("\n📎 Filas duplicadas:")
    duplicadas = df.duplicated().sum()
    if duplicadas > 0:
        print(f"🔴 Hay {duplicadas} filas duplicadas.")
        display(df[df.duplicated()])
    else:
        print("✅ No hay filas duplicadas.")

    # 4. Columnas duplicadas
    print("\n📎 Columnas duplicadas:")
    columnas_duplicadas = df.T.duplicated().sum()
    if columnas_duplicadas > 0:
        print(f"🔴 Hay {columnas_duplicadas} columnas duplicadas.")
    else:
        print("✅ No hay columnas duplicadas.")




#------------------------------------------------FUNCIONES MODELO------------------------------------------------------------

def guardar_grafico(nombre_archivo: str, carpeta: str):
    """
    Guarda el gráfico actual como archivo PNG en la carpeta especificada.

    Parámetros:
    - nombre_archivo (str): nombre base del archivo (sin extensión).
    - carpeta (str): ruta de la carpeta donde guardar la imagen.
    """
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    ruta = os.path.join(carpeta, f"{nombre_archivo}.png")
    plt.savefig(ruta, bbox_inches="tight")
    print(f"✅ Gráfico guardado en: {ruta}")


def analizar_target_abandono(df: pd.DataFrame, target_col: str = 'Abandono', figsize: Tuple[int, int] = (10, 5),
                             style: str = 'whitegrid', context: str = 'notebook') -> None:

    """
    Muestra un análisis simple del target (churn), incluyendo conteo, proporciones y un gráfico.
    
    Parámetros:
    - df: DataFrame de entrada.
    - target_col: nombre de la columna objetivo (por defecto: 'Abandono').
    - figsize: tamaño de la figura (tupla).
    - style: estilo de seaborn (por defecto: 'whitegrid').
    - context: contexto de seaborn (por defecto: 'notebook').
    """
    
    # Estilo visual
    sns.set(style=style, context=context)
    plt.rcParams['figure.figsize'] = figsize
    
    # Conteo y proporciones
    print('🔹 Target counts:')
    print(df[target_col].value_counts(dropna=False))
    
    print('\n🔹 Target proportion:')
    print(df[target_col].value_counts(normalize=True))
    
    # Gráfico
    sns.countplot(data=df, x=target_col, palette='pastel')
    plt.title('Distribución del Target (Abandono)', fontsize=14)
    plt.xlabel(target_col)
    plt.ylabel('Frecuencia')
    plt.show()

def calcular_correlaciones_abandono(df: pd.DataFrame, target_col: str = 'Abandono', top_n: Optional[int] = None) -> pd.Series:

    """
    Calcula las correlaciones de todas las variables numéricas con el target.

    Parámetros:
    - df: DataFrame de entrada.
    - target_col: columna objetivo (por defecto: 'Abandono').
    - top_n: número de variables con mayor correlación a mostrar (si se desea limitar).

    Retorna:
    - Series con correlaciones ordenadas descendente (más correladas arriba).
    """
    # Asegurar que el target esté en formato numérico
    df[target_col] = df[target_col].astype(int)

    # Seleccionar columnas numéricas (incluye el target)
    df_num = df.select_dtypes(include=['number'])

    # Calcular correlaciones con el target
    correlaciones = df_num.corr()[target_col].sort_values(ascending=False)

    # Mostrar top_n si se desea
    if top_n is not None:
        correlaciones = correlaciones.head(top_n)

    print(f"🔹 Correlaciones con '{target_col}':")
    print(correlaciones)

    return correlaciones

def calcular_correlacion_bool(df: pd.DataFrame, objetivo: str = 'Abandono') -> pd.Series:
    """
    Calcula la correlación de variables booleanas con la variable objetivo.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con las columnas booleanas y la variable objetivo.
        objetivo (str): Nombre de la variable objetivo (por defecto 'abandono').
    
    Retorna:
        pd.Series: Correlación de cada variable booleana con la variable objetivo,
                   ordenada de mayor a menor, sin incluir la variable objetivo.
    """
    # Seleccionar columnas booleanas
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    # Asegurarse de que la variable objetivo esté incluida
    if objetivo not in bool_cols:
        bool_cols.append(objetivo)
    
    # Convertir a int (0/1)
    df_corr = df[bool_cols].astype(int)

    # Calcular correlación y ordenar
    correlation = df_corr.corr()[objetivo].drop(objetivo).sort_values(ascending=False)
    return correlation

# ======================================
# 1. SEPARACIÓN BALANCEADA DEL 20% FINAL
# ======================================
 

def separacion_df_inferencia_test_final(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Separa un DataFrame en un conjunto de validación balanceado (20% del total)
    y un conjunto de entrenamiento (80% restante).

    La validación contiene 10% de muestras de cada clase (0 y 1) para mantener balance.

    Parámetros:
    - df (pd.DataFrame): DataFrame original que debe contener la columna 'Abandono'.

    Retorna:
    - df_validacion (pd.DataFrame): conjunto balanceado para validación.
    - df_train (pd.DataFrame): conjunto para entrenamiento.
    """

    # Separar por clase
    df_churn_0 = df[df['Abandono'] == 0]
    df_churn_1 = df[df['Abandono'] == 1]

    # Cantidad del 10% de cada clase
    n_10pct_0 = int(0.10 * len(df))
    n_10pct_1 = int(0.10 * len(df))

    # Sample aleatorio (sin reemplazo)
    valid_0 = df_churn_0.sample(n=n_10pct_0, random_state=42)
    valid_1 = df_churn_1.sample(n=n_10pct_1, random_state=42)

    # Concatenar para tener el 20% de validación balanceado
    df_validacion = pd.concat([valid_0, valid_1]).reset_index(drop=True)

    # # Crear conjunto de entrenamiento excluyendo los de validación
    df_train = df.drop(df_validacion.index).reset_index(drop=True)

    return df_validacion, df_train

def aplicacion_modelo(X: pd.DataFrame, y: pd.Series, carpeta_imagenes: str) -> Tuple:
    """
    Realiza el entrenamiento, optimización (GridSearchCV), evaluación y comparación
    de varios modelos clásicos para clasificación.

    Parámetros:
    - X (pd.DataFrame o np.ndarray): variables predictoras.
    - y (pd.Series o np.ndarray): variable objetivo binaria.

    Retorna:
    - best_model: mejor estimador entrenado tras optimización.
    - scaler: objeto StandardScaler utilizado para escalar X.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==============================
    # 3. MODELOS Y PARÁMETROS

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    param_grids = {
        "Regresión Logística": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"]},
        "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    }

    # ==============================
    # 4. ENTRENAMIENTO Y EVALUACIÓN

    resultados = []

    plt.figure(figsize=(10, 8))

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Optimizando {nombre}...")
        start = time.time()

        grid = GridSearchCV(
            estimator=modelo,
            param_grid=param_grids[nombre],
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train_scaled, y_train)
        tiempo = round(time.time() - start, 2)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

        # Métricas
        cm = confusion_matrix(y_test, y_pred)

        if y_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')
        else:
            roc_auc = None

        resultados.append({
            "Modelo": nombre,
            "Mejores Parámetros": grid.best_params_,
            "AUC": roc_auc,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "Matriz Confusión": cm,
            "Tiempo (s)": tiempo,
            "Best_Model": best_model
        })

    # ==============================
    # 5. RESULTADOS EN DATAFRAME

    df_resultados = pd.DataFrame(resultados).sort_values(by="AUC", ascending=False)
    display(df_resultados)

    # ==============================
    # 6. CURVA ROC COMPARATIVA

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC de modelos")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(carpeta_imagenes, "curva_ROC_modelos.png"))
    plt.show()

    # ==============================
    # 7. MEJOR MODELO

    mejor_modelo = df_resultados.iloc[0]
    best_model = mejor_modelo["Best_Model"]

    print("\n🏆 Mejor modelo:")
    print("Modelo:", mejor_modelo["Modelo"])
    print("AUC:", round(mejor_modelo["AUC"], 3))
    print("Accuracy:", round(mejor_modelo["Accuracy"], 3))
    print("Recall:", round(mejor_modelo["Recall"], 3))
    print("F1:", round(mejor_modelo["F1"], 3))
    print("Mejores Parámetros:", mejor_modelo["Mejores Parámetros"])
    print("Matriz de Confusión:\n", mejor_modelo["Matriz Confusión"])

    # ==============================
    # 8. IMPORTANCIA DE VARIABLES (si aplica)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            'Variable': X_train.columns,
            'Importancia': best_model.feature_importances_
        }).sort_values(by='Importancia', ascending=False)

        print("\n🔥 Importancia de variables del mejor modelo:")
        display(importances)

        plt.figure(figsize=(10, 6))
        plt.barh(importances['Variable'], importances['Importancia'])
        plt.gca().invert_yaxis()
        plt.title(f"Top variables - {mejor_modelo['Modelo']}")
        guardar_grafico(f"importancia_variables_{mejor_modelo['Modelo'].replace(' ', '_')}", carpeta_imagenes)

        plt.show()
    else:
        print("\n⚠️ Este modelo no tiene atributo 'feature_importances_'")
    
    return best_model, scaler

def validacion_inferencia_testfinal(X: pd.DataFrame, scaler, best_model, df_validacion: pd.DataFrame, carpeta_imagenes: str) -> Optional[np.ndarray]:

    """
    Realiza la validación del modelo final usando un conjunto de validación separado,
    escalando las variables, haciendo predicciones y mostrando métricas y curva ROC.

    Parámetros:
    - X (pd.DataFrame): DataFrame con las columnas predictoras utilizadas para entrenamiento.
    - scaler (StandardScaler): objeto scaler ajustado para escalar las variables predictoras.
    - best_model: modelo entrenado (estimador sklearn) a usar para inferencia.
    - df_validacion (pd.DataFrame): conjunto de validación con las mismas columnas y variable 'Abandono'.

    Retorna:
    - y_val_prob (np.ndarray): probabilidades de la clase positiva para el conjunto de validación.
      Retorna None si ocurre algún error en el proceso.
    """

       
    try:
        X_val = df_validacion[X.columns]
    except Exception as e:
        print("Error al crear X_val:", e)
        return None

    try:
        y_val = df_validacion['Abandono']
    except Exception as e:
        print("Error al crear y_val:", e)
        return None

  
    try:
        X_val_scaled = scaler.transform(X_val)
    except Exception as e:
        print("Error al escalar:", e)
        return None

   
    try:
        y_val_pred = best_model.predict(X_val_scaled)
        y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]
     
    except Exception as e:
        print("Error al predecir:", e)
        return None

  
    try:
        print("Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Recall:", recall_score(y_val, y_val_pred))
        print("F1 Score:", f1_score(y_val, y_val_pred))
        print("Matriz de Confusión:\n", confusion_matrix(y_val, y_val_pred))

        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
        roc_auc_val = auc(fpr_val, tpr_val)
        print("AUC (Inferencia Test):", round(roc_auc_val, 3))

        plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Curva ROC - Inferencia Test final")
        plt.legend()
        plt.savefig(os.path.join(carpeta_imagenes, "curva_ROC_inferencia_test"))
        plt.show()
    except Exception as e:
        print("Error en métricas o plot:", e)
        return None

    return y_val_prob


def clasificar_riesgo(df_validation: pd.DataFrame, y_val_prob: np.ndarray) -> pd.DataFrame:
    """
    Añade a un DataFrame de validación la probabilidad de abandono y clasifica en niveles de riesgo.

    Parámetros:
    - df_validation (pd.DataFrame): DataFrame con datos de validación.
    - y_val_prob (np.ndarray): probabilidades predichas de abandono para cada fila en df_validation.

    Retorna:
    - df_validation (pd.DataFrame): DataFrame original enriquecido con columnas:
        - 'Prob_Abandono': probabilidad de abandono.
        - 'Nivel_Riesgo': categorización en niveles de riesgo ('Muy Bajo', 'Bajo', etc.).
    """

    # Añadir columna de probabilidad de abandono
    df_validation["Prob_Abandono"] = y_val_prob

    # Crear niveles de riesgo
    df_validation["Nivel_Riesgo"] = pd.cut(df_validation["Prob_Abandono"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],  include_lowest=True)
    return df_validation

def guardar_resultados(df: pd.DataFrame, nombre_modelo: str):
    """
    Guarda el DataFrame resultante de un modelo en un archivo CSV y lo imprime por consola.

    Args:
        df (pd.DataFrame): DataFrame de validación.
        nombre_modelo (str): Nombre identificativo del modelo (ej. 'modelo1').
    """
    print(f"\n✅ ¡{nombre_modelo.upper()} ENTRENADO CON ÉXITO!")

    print("\n📊 Mostrando dataframe resultante:")
    print(df)

    ruta_archivo = f"data/resultados_{nombre_modelo.lower()}.csv"
    df.to_csv(ruta_archivo, index=False)

    print(f"\n💾 Guardando resultados en: {ruta_archivo}")
    print(f"\n✅ ¡RESULTADOS DE {nombre_modelo.upper()} GUARDADOS CON ÉXITO!\n")
