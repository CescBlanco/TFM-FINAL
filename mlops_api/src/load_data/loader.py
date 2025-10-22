import pandas as pd
from src.config import DATA_PATH, VALIDATION_OUTPUT_PATH

def cargar_datos(nombre_experimento, features):
    df = pd.read_csv(DATA_PATH)
    df = df[df['Edad'] >= 18].reset_index(drop=True)

    df = df.rename(columns={
        'EsChurn': 'Abandono',
        'DiaFav_miércoles': 'DiaFav_miercoles',
        'DiaFav_sábado': 'DiaFav_sabado'
    })

    def separacion_df_inferencia(df):
        df_0 = df[df['Abandono'] == 0]
        df_1 = df[df['Abandono'] == 1]
        n = int(0.10 * len(df))
        valid_0 = df_0.sample(n=n, random_state=42)
        valid_1 = df_1.sample(n=n, random_state=42)
        df_valid = pd.concat([valid_0, valid_1]).reset_index(drop=True)
        df_train = df.drop(df_valid.index).reset_index(drop=True)
        return df_valid, df_train

    df_valid, df_train = separacion_df_inferencia(df)
    df_valid.to_csv(f"{VALIDATION_OUTPUT_PATH}/df_validacion_{nombre_experimento}.csv", index=False)

    return df_train[features], df_train['Abandono']