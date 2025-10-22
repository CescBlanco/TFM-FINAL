import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], path: str) -> None:
    """
    Dibuja y guarda la matriz de confusión como una imagen.

    Parámetros:
        y_true (List[int]): Lista de las etiquetas verdaderas (valores reales de la clase).
        y_pred (List[int]): Lista de las etiquetas predichas por el modelo (valores predichos).
        path (str): Ruta donde se guardará la imagen de la matriz de confusión.

    Retorna:
        None: La función guarda la matriz de confusión como una imagen en la ruta especificada.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()