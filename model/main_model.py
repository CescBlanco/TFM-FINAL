from model import modelo
from model.funciones_modelo import *
from sklearn.pipeline import Pipeline
from joblib import dump


def main_model():
    print("\nÚltima preparación para el entrenamiento de modelos...\n")

    # Entrenamiento de los tres modelos y obtención de los resultados
    (df_validacion_modelo1, df_validacion_modelo2, df_validacion_modelo3),(model1, scaler1, model2, scaler2, model3, scaler3, X_1,X_2, X_3) = modelo.run()
    
    # Guardado y muestra de resultados de cada modelo
    guardar_resultados(df_validacion_modelo1, "modelo 1")
    guardar_resultados(df_validacion_modelo2, "modelo 2")
    guardar_resultados(df_validacion_modelo3, "modelo 3")

    

    # Crear un pipeline para el modelo 3
    pipeline_final = Pipeline([
        ("scaler", scaler3),
        ("modelo", model3)
    ])
    
    # Guardar el pipeline como archivo .joblib
    dump(pipeline_final, "model/api_pickle/pipeline_abandono.joblib")
    print("\n✅ Modelo 3 guardado como pipeline_abandono.joblib")

    # Guardar columnas usadas por el modelo 3
    columnas_modelo3 = X_3.columns.tolist()

    with open("model/api_pickle/columnas_modelo3.txt", "w") as f:
        for col in columnas_modelo3:
            f.write(col + "\n")

    print("📝 Columnas del modelo guardadas en columnas_modelo3.txt")


    print("\n🎉 ¡Entrenamiento y guardado completados!\n")

if __name__ == "__main__":
    main_model()