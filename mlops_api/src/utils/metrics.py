from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def calcular_metricas(y_true, y_pred, y_prob):
    """Calcula las métricas más comunes y devuelve un diccionario."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }