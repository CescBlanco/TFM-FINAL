import streamlit as st
import pandas as pd
import requests
import json
import time
import mlflow
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow import *

LOGO_GYM= "streamlit/assets/cem_horta-removebg-preview.png"
LOGO_AYUNTAMIENTO = "streamlit/assets/LOGO-AJUNTAMENT.png"

# URL de tu API local
API_URL = "http://localhost:8000"


BOOL_COL = ["Sexo_Mujer", "UsoServiciosExtra", "TienePagos", "TieneAccesos",
                    "DiaFav_domingo", "DiaFav_jueves", "DiaFav_lunes", "DiaFav_martes",
                    "DiaFav_miercoles", "DiaFav_sabado", "DiaFav_viernes",
                    "EstFav_invierno", "EstFav_otono", "EstFav_primavera", "EstFav_verano"
                ]


NAME_EXPERIMENT_3= 'Experimento_v3'   
NAME_EXPERIMENT_2 = 'Experimento_v2'
NAME_EXPERIMENT_1 = 'Experimento_v1'
METRIC= 'auc'


#RUN IDs para las validaciones del experimento 1 y 2.
RUN_ID_INF_1= '7ec94007ba584b68b695afa7e79825cc'
RUN_ID_INF_2= '217f131f5f8246d0b56d201738790051'     
RUN_ID_INF_3 = "73233a8103ba4517bdd5f7f9b4b2576e"  #RunID pegado despuès de encontrarlo en MLFLow IU (experimento3)



#Ruta donde se guardaran cada artefacto segun el experimento 1 y 2 (no usados)
FOLDER_DESTINO_1 = 'mlops_api/data_mlops_api/inferencia_predicciones_exp1'
FOLDER_DESTINO_2 = 'mlops_api/data_mlops_api/inferencia_predicciones_exp2'
# Ruta donde se guardarán los artefactos descargados de la inferencia 3 (la importante)
FOLDER_DESTINO_3= 'mlops_api/data_mlops_api/inferencia_predicciones_exp3'

def cargar_columnas_modelo(path):
    try:
        with open(path, 'r') as f:
            columnas = f.read().splitlines()
        return columnas
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo: {path}")
        return []

COLUMNAS_MODELO = cargar_columnas_modelo('mlops_api/src/api/columnas_modelo3.txt')
# Función para manejar las peticiones a la API
def obtener_predicciones_api(endpoint, data):
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        response.raise_for_status()  # Esto lanzará una excepción si la respuesta es 4xx o 5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error: {e}")
        return None

# Función para formulario individual
def input_userdata():
    # Usamos columnas para organizar el formulario
    col1, col2 = st.columns([1, 1])

    with col1:
        # Títulos de las secciones
        st.markdown("<h3 style='color: #888;'>Información Personal</h3>", unsafe_allow_html=True)   
        
        Edad = st.number_input("Edad", min_value=18, max_value=120, value=30)
        Sexo_Mujer = st.checkbox("Sexo Mujer")
        TienePagos = st.checkbox("Tiene Pagos")
        TieneAccesos = st.checkbox("Tiene Accesos")
        
    with col2:
        st.markdown("<h3 style='color: #888;'>Visitas y Actividad</h3>", unsafe_allow_html=True)
          
        TotalVisitas = st.number_input("Total Visitas", min_value=0, value=0)
        DiasActivo = st.number_input("Días Activo", min_value=0, value=0)
        VisitasUlt90 = st.number_input("Visitas Últimos 90 días", min_value=0, value=0)
        VisitasUlt180 = st.number_input("Visitas Últimos 180 días", min_value=0, value=0)
        VisitasPrimerTrimestre = st.number_input("Visitas Primer Trimestre", min_value=0, value=0)
        VisitasUltimoTrimestre = st.number_input("Visitas Último Trimestre", min_value=0, value=0)

    st.markdown("---")  # Línea separadora para organización visual

    st.markdown("<h3 style='color: #888;'>Preferencias y Estilo de Vida</h3>", unsafe_allow_html=True)
    
    # Usamos columnas de nuevo para agrupar preferencias
    col3, col4 = st.columns([1, 1])
    
    with col3:
        EstFav_invierno = st.checkbox("Estación Favorita Invierno")
        EstFav_otono = st.checkbox("Estación Favorita Otoño")
        EstFav_primavera = st.checkbox("Estación Favorita Primavera")
        EstFav_verano = st.checkbox("Estación Favorita Verano")
        
    with col4:
        DiaFav_domingo = st.checkbox("Día Favorito Domingo")
        DiaFav_jueves = st.checkbox("Día Favorito Jueves")
        DiaFav_lunes = st.checkbox("Día Favorito Lunes")
        DiaFav_martes = st.checkbox("Día Favorito Martes")
        DiaFav_miercoles = st.checkbox("Día Favorito Miércoles")
        DiaFav_sabado = st.checkbox("Día Favorito Sábado")
        DiaFav_viernes = st.checkbox("Día Favorito Viernes")

    st.markdown("---")  # Línea separadora

    st.markdown("<h3 style='color: #888;'>Ratio y Diversidad de Servicios</h3>", unsafe_allow_html=True)

    # Agrupamos en una columna
    col5, col6 = st.columns([1, 1])

    with col5:
        UsoServiciosExtra = st.checkbox("Uso Servicios Extra")
        
    with col6:
        ratio_cantidad_2025_2024 = st.number_input("Ratio cantidad 2025/2024", value=1.0, format="%.3f")
        Diversidad_servicios_extra = st.number_input("Diversidad servicios extra", min_value=0, max_value=100, value=1)


    return {
        "Edad": Edad,
        "Sexo_Mujer": Sexo_Mujer,
        "UsoServiciosExtra": UsoServiciosExtra,
        "ratio_cantidad_2025_2024": ratio_cantidad_2025_2024,
        "Diversidad_servicios_extra": Diversidad_servicios_extra,
        "TienePagos": TienePagos,
        "TotalVisitas": TotalVisitas,
        "DiasActivo": DiasActivo,
        "VisitasUlt90": VisitasUlt90,
        "VisitasUlt180": VisitasUlt180,
        "TieneAccesos": TieneAccesos,
        "VisitasPrimerTrimestre": VisitasPrimerTrimestre,
        "VisitasUltimoTrimestre": VisitasUltimoTrimestre,
        "DiaFav_domingo": DiaFav_domingo,
        "DiaFav_jueves": DiaFav_jueves,
        "DiaFav_lunes": DiaFav_lunes,
        "DiaFav_martes": DiaFav_martes,
        "DiaFav_miercoles": DiaFav_miercoles,
        "DiaFav_sabado": DiaFav_sabado,
        "DiaFav_viernes": DiaFav_viernes,
        "EstFav_invierno": EstFav_invierno,
        "EstFav_otono": EstFav_otono,
        "EstFav_primavera": EstFav_primavera,
        "EstFav_verano": EstFav_verano,
        
    }

def encontrar_metricas_experimento(NAME_EXPERIMENT, metric= 'auc'):

        #Extraer metricas del modelo del experiemento 3 
        client = MlflowClient()
        experiment = client.get_experiment_by_name(NAME_EXPERIMENT)
        if not experiment:
            raise ValueError(f"No se encontró el experimento {NAME_EXPERIMENT} en MLflow")

        best_run = client.search_runs(
            [experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )[0]
        run_id_exp3 = best_run.info.run_id
        
        model = mlflow.sklearn.load_model(f"runs:/{run_id_exp3}/model")

        run_exp3= mlflow.get_run(run_id_exp3)
        metrics_exp3= run_exp3.data.metrics  # Esto es un diccionario de métricas
  
        # Si deseas acceder a alguna métrica específica, por ejemplo 'auc':
        auc_exp3 = round(metrics_exp3.get('auc', None),2)

        accuracy_exp3 = round(metrics_exp3.get('accuracy', None),2)  # Retorna None si no se encuentra
        f1_exp3 = round(metrics_exp3.get('f1_score', None),2)
        recall_exp3 = round(metrics_exp3.get('recall', None),2)

        return auc_exp3, accuracy_exp3, f1_exp3, recall_exp3

def encontrar_csv_inferencias(NAME_EXPERIMENT, folder_destino_ex, run_id):
    # Especifica el run_id conocido
    if not os.path.exists(folder_destino_ex):
        os.makedirs(folder_destino_ex)

    try:
        # Obtener el run para obtener el artifact_uri
        run_inf = mlflow.get_run(run_id)
        # Mostrar el URI de artefactos para saber a qué carpeta acceder
        artifact_uri = run_inf.info.artifact_uri

        # Descargar los artefactos usando el URI obtenido directamente
        artifact_path = mlflow.artifacts.download_artifacts(artifact_uri)

        # Verificar si los artefactos fueron descargados correctamente
        if not os.path.exists(artifact_path):
            st.error(f"No se encontraron artefactos en el path: {artifact_path}")
            return None, None, None

    except Exception as e:
        st.error(f"Error al descargar los artefactos: {e}")
        return None, None, None

    # Si los artefactos fueron descargados correctamente
    archivos_descargados = os.listdir(artifact_path)

    if archivos_descargados:
        for archivo in archivos_descargados:
            archivo_origen = os.path.join(artifact_path, archivo)
            archivo_destino = os.path.join(folder_destino_ex, archivo)

            # Copiar el archivo
            shutil.copy(archivo_origen, archivo_destino)
    else:
        st.write("No se encontraron archivos en el directorio de artefactos.")

    # Mostrar los archivos en la carpeta destino
    archivos_guardados = os.listdir(folder_destino_ex)
    archivos_csv = [archivo for archivo in archivos_guardados if archivo.endswith('.csv')]

    if archivos_csv:
        dataframes = {}
        for archivo_csv in archivos_csv:
            nombre_variable = archivo_csv.replace('.csv', '')
            ruta_completa = os.path.join(folder_destino_ex, archivo_csv)

            try:
                # Cargar el archivo CSV en un DataFrame
                df = pd.read_csv(ruta_completa)
                dataframes[nombre_variable] = df
            except Exception as e:
                st.error(f"Error al cargar el archivo {archivo_csv}: {e}")
    else:
        st.write("No se encontraron archivos CSV en la carpeta de destino.")

    # Verificar si las claves existen en el diccionario antes de acceder a ellas
    df_archivo_global = dataframes.get(f'importancias_global_{NAME_EXPERIMENT}', None)
    df_archivo_persona = dataframes.get(f'importancias_persona_{NAME_EXPERIMENT}', None)
    df_archivo_preds = dataframes.get(f'preds_{NAME_EXPERIMENT}', None)

    # Devolver los dataframes
    return df_archivo_global, df_archivo_persona, df_archivo_preds      


def encontrar_metricas_inferencia(run_id):
    # Obtener el run para obtener el artifact_uri
    run_inf = mlflow.get_run(run_id)
    # Obtener las métricas del run_inf
    metrics = run_inf.data.metrics  # Esto es un diccionario de métricas


    metricas_dict = {}
    for metric_name, metric_value in metrics.items():
        # Convertir el nombre de la métrica para que sea una variable válida (sin el prefijo 'val_')
        if metric_name.startswith('val_'):
            variable_name = metric_name[4:]  # Eliminar el prefijo 'val_' para que la variable se llame 'accuracy' en lugar de 'val_accuracy'
        else:
            variable_name = metric_name
        metricas_dict[variable_name] = metric_value
    # Crear una variable con el nombre limpio y asignar el valor correspondiente
    accuracy = round(metricas_dict.get('accuracy', None),2)  # Retorna None si no se encuentra
    auc= round(metricas_dict.get('auc', None),2)  
    f1 = round(metricas_dict.get('f1', None),2)  
    recall = round(metricas_dict.get('recall', None),2)  

    return accuracy, auc, f1, recall

def plot_importancias(df_global ):

    fig, ax = plt.subplots(figsize=(10, 6))
    df_archivo_global= df_global.sort_values(by='Importance', ascending= False)
    ax.barh(df_archivo_global['Feature'], df_archivo_global['Importance'])
    ax.invert_yaxis()  # Invertir el eje Y para que la variable más importante esté arriba

    # Añadir el valor de la importancia a cada barra
    for index, value in enumerate(df_archivo_global['Importance']):
        ax.text(value, index, f'{value:.4f}', va='center', ha='left', size=6, color='white', fontweight='bold')

    # Configurar fondo transparente
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    ax.set_xlabel("Importancia", color='white')
    ax.set_title("Top variables por importancia", color='white')
    ax.tick_params(axis='both', colors='white')
    return fig

def plots_experimentos_sinuso(df, variable_importante): 
    # Filtrar los datos por cada categoría de Abandono
    abandono_0 = df[df['EsChurn'] == 0][variable_importante]
    abandono_1 = df[df['EsChurn'] == 1][variable_importante]

    # Crear el histograma en un solo gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(abandono_0, bins=50, alpha=0.6, label='abandono 0', color='blue')
    ax.hist(abandono_1, bins=50, alpha=0.6, label='abandono 1', color='red')

    ax.set_xlabel(variable_importante)
    ax.set_ylabel('Cantidad de Clientes')
    ax.set_title(f'Distribución de {variable_importante} por abandono')
    ax.legend()
    ax.grid(True)
    return fig

def piechart_edad(df): 
    nivel_riesgo_counts = df["nivel_riesgo"].value_counts().reindex(["Muy bajo", "Bajo", "Medio", "Alto", "Muy Alto"])          
    # Crear el gráfico de torta (pie chart)
    fig, ax = plt.subplots(figsize=(6, 6))
    nivel_riesgo_counts.plot(
        kind="pie",
        autopct="%1.1f%%",      # Mostrar porcentaje con un decimal
        ax=ax,
        colors=sns.color_palette("Greens", 5),  # Paleta de colores
        startangle=90,          # Rotar el gráfico para que empiece desde un ángulo de 90 grados
        legend=False ,           # No mostrar la leyenda
        labels=nivel_riesgo_counts.index  # Asegúrate de que las etiquetas estén correctas
    
    )
    # Personalizar el gráfico
    ax.set_ylabel("", color='white')  # Quitar la etiqueta en el eje y
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    # Cambiar color de las etiquetas a blanco
    for label in ax.texts:
        label.set_color('white')
    for i, label in enumerate(ax.texts):
        if '%' in label.get_text():  # Solo aplicar el cambio a las etiquetas de porcentaje
            # Establecer el color de la etiqueta de porcentaje a blanco
            label.set_bbox(dict(facecolor='black', alpha=0.7, edgecolor='none'))  # Fondo negro semi-transparente
    return fig

def tabla_recuento_resultados(df):
    # Crear la columna de estado con base en y_true
    df['estado'] = df['y_true'].map({False: 'Activo', True: 'Abandonado'})

    # Asegurarse de que 'nivel_riesgo' esté en el orden correcto
    orden_riesgo = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    df['nivel_riesgo'] = pd.Categorical(df['nivel_riesgo'], categories=orden_riesgo, ordered=True)

    # Filtrar datos por estado
    df_activos = df[df['estado'] == 'Activo']
    df_abandonados = df[df['estado'] == 'Abandonado']

    # Agrupar por 'nivel_riesgo' y calcular las métricas para "Activo"
    grouped_activos = df_activos.groupby('nivel_riesgo').agg(
        abonados=('y_true', 'size'),  # Recuento de abonados por grupo
        promedio_probabilidad=('y_prob', 'mean')  # Promedio de probabilidad de abandono por grupo
    ).reset_index()

    # Agrupar por 'nivel_riesgo' y calcular las métricas para "Abandonado"
    grouped_abandonados = df_abandonados.groupby('nivel_riesgo').agg(
        abonados=('y_true', 'size'),  # Recuento de abonados por grupo
        promedio_probabilidad=('y_prob', 'mean')  # Promedio de probabilidad de abandono por grupo
    ).reset_index()

    # Renombrar las columnas para hacerlas más comprensibles
    grouped_activos.rename(columns={
        'nivel_riesgo': 'Grupo de Riesgo',
        'abonados': 'Nº Clientes Abandonados',
        'promedio_probabilidad': 'Promedio % de Abandono'
    }, inplace=True)

    grouped_abandonados.rename(columns={
        'nivel_riesgo': 'Grupo de Riesgo',
        'abonados': 'Nº Clientes Abandonados',
        'promedio_probabilidad': 'Promedio % de Abandono'
    }, inplace=True)

    # Eliminar el índice de las tablas antes de pasarlas a st.table()
    grouped_activos_reset = grouped_activos.reset_index(drop=True)
    grouped_abandonados_reset = grouped_abandonados.reset_index(drop=True)
    return grouped_activos_reset, grouped_abandonados_reset

def categorizacion_variables_importancia(df):     
     # Crear grupos de edad
    df["GrupoEdad"] = pd.cut(df["Edad_inicial"],bins=[18, 25, 35, 45, 55, 65, 80, df["Edad_inicial"].max()],
                    labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66–80", "80+"],include_lowest=True)               
    # Dividir TotalVisitas en cuartiles
    df['TotalVisitas_categoria'] = pd.qcut(df['TotalVisitas_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir DiasActivo en cuartiles
    df['DiasActivo_categoria'] = pd.qcut(df['DiasActivo_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])
    
    # Dividir VisitasUlt90 en cuartiles
    df['VisitasUlt90_categoria'] = pd.qcut(df['VisitasUlt90_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir VisitasUlt180 en cuartiles
    df['VisitasUlt180_categoria'] = pd.qcut(df['VisitasUlt180_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])
                
    # Dividir VisitasPrimerTrimestre en cuartiles
    df['VisitasPrimerTrimestre_categoria'] = pd.qcut(df['VisitasPrimerTrimestre_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir VisitasUltimoTrimestre en cuartiles
    df['VisitasUltimoTrimestre_categoria'] = pd.qcut(df['VisitasUltimoTrimestre_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])
    return df

def box_plot(df, variable_x, variable_y, x_label):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Establecer los parámetros de los valores atípicos
    flierprops = dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none')
    sns.boxplot(x=variable_x, y=variable_y, data=df, palette='Greens', color='white',
                linewidth=2.8,linecolor='grey', flierprops=flierprops, ax=ax)
    #plt.title(title, color= 'white')
    plt.xlabel(x_label, color= 'white')
    plt.ylabel("Probabilidad de abandono", color='white')
    plt.grid(True, color= 'white')
                # Configurar fondo transparente
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    ax.tick_params(axis='both', colors='white')
    return fig

# Función para mostrar gráfico e interpretación
def mostrar_grafico_y_descripcion(eleccion, df):
    if eleccion == "Probabilidad de Abandono por Grupos de Edad":
        # Aquí generas tu gráfico para el primer caso
        fig_edad = box_plot(df, "GrupoEdad", "y_prob", "Grupos de Edad")
        st.pyplot(fig_edad)
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:

            - **`Jóvenes (18-35 años)`**: Mayor probabilidad de abandono.
            - **`Adultos mayores (66-80 y 80+)`**: Menor probabilidad de abandono.
            - **`Outliers`**: Grupo de 66-80 tiene comportamientos extremos de abandono.

            **💼📈 Interpretación para el Negocio**:

            - **`Jóvenes`**: Se deben implementar estrategias de retención específicas para este segmento (mejorar experiencia, promociones, etc.).
            - **`Mayores`**: Los usuarios de más edad parecen más comprometidos; mantener y mejorar la retención de este grupo es clave.
            """)

            
    
    elif eleccion == "Probabilidad de Abandono por Grupos de Días Activos":
        
        fig_activos = box_plot(df, "DiasActivo_categoria", "y_prob", "Grupos de dias activos")
        st.pyplot(fig_activos)
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
            
            - **`Usuarios Activos`**:  Menor probabilidad de abandono.
            - **`Outliers`**:  Algunos usuarios muy activos todavía tienen alta probabilidad de abandono.

            
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Más actividad`**: Fomentar la actividad continua reduce el abandono.
            - **`Estrategia`**: Enfocar recursos en mantener a los usuarios activos (notificaciones, recompensas, ofertas).
            """)
        
        
    elif eleccion == "Probabilidad de Abandono por Grupos de Visitas Últimos 180 Días":
        fig_ultim180 = box_plot(df, "VisitasUlt180_categoria", "y_prob", "Grupos de visitas últimos 180 días")
        st.pyplot(fig_ultim180)

        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con más visitas`**: Menor probabilidad de abandono.
            - **`Usuarios con pocas visitas`**: Mayor probabilidad de abandono.
            
            
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Los outliers`** también son notables, especialmente en el grupo "Alto", lo que sugiere que, aunque un número considerable de usuarios con muchas visitas tiene una baja probabilidad de abandono, algunos casos se comportan de manera diferente.
            """)
        
    elif eleccion == "Probabilidad de Abandono por Visitas Primer Trimestre":
        fig_primertrim = box_plot(df, "VisitasPrimerTrimestre_categoria", "y_prob", "Grupo de Visitas Primer Trimestre")
        st.pyplot(fig_primertrim)

        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con más visitas en el primer trimestre`**: Menor probabilidad de abandono.
            - **`Outliers`**: Algunos usuarios con muchas visitas siguen teniendo una alta probabilidad de abandono. Casos fuera del comportamiento (distribución de los datos)         
                
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Visitas tempranas`**: Las visitas durante el primer trimestre son un buen predictor de retención a largo plazo.
            - **`Estrategia`**: Incentivar visitas frecuentes en los primeros meses (promociones para nuevos usuarios).
            """)
    elif eleccion =="Probabilidad de Abandono por Estación Favorita Otoño":
        fig_otoño = box_plot(df, "EstFav_otono_inicial", "y_prob", "Estación favorita Otoño")
        st.pyplot(fig_otoño) 

        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con Otoño como estación favorita`**: Levelemente más alta probabilidad de abandono.
            - **`Outliers`**: Algunas desviaciones de comportamiento entre usuarios con y sin Otoño como favorita.         
                
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Estación menos significativa`**: La estación favorita tiene un impacto menor en la retención.
            - **`Oportunidad de segmentación`**: A pesar de su menor impacto, podría usarse para campañas personalizadas o promociones estacionales. 
            """)
    
    elif eleccion =="Probabilidad de Abandono por si Tiene Pagos":
        fig_pagos = box_plot(df, "TienePagos_inicial", "y_prob", "Tiene Pagos")
        st.pyplot(fig_pagos)

        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
            
            - **`Usuarios que pagan`**:  Mucha menor probabilidad de abandono.
            - **`Usuarios que no pagan`**:  Alta probabilidad de abandono (casi 80%).
            
            **💼📈 Interpretación para el Negocio**:
    
            - **`Usuarios pagos`**:Son mucho más valiosos y están más comprometidos.
            - **`Estrategia`**: Focalizarse en convertir usuarios gratuitos a pagos y retener a los clientes de pago a través de una mejor experiencia, soporte personalizado, y programas de fidelización.
            """)

# Diccionario de estrategias
ESTRATEGIAS_FIDELIZACION = {
    "Muy Bajo": ["""
        1. **`Programa de recompensas por uso continuo`**: Implementar un sistema de puntos para los usuarios frecuentes, que puedan canjear por descuentos, contenido exclusivo o productos premium.
        2. **`Acceso anticipado a nuevas funcionalidades`**: Los usuarios más activos y pagos pueden ser invitados a probar nuevas funciones antes que el resto de los usuarios. Esto crea un sentido de exclusividad.
        3. **`Beneficios por referencia`**: Ofrecer recompensas por recomendar la plataforma a amigos o colegas. Esto podría ser un mes gratis o un descuento para ambos (referente y referido).
        4. **`Ofertas personalizadas para el perfil de uso`**: Ofrecer descuentos o beneficios exclusivos basados en el comportamiento de uso del cliente. Ejemplo: si un usuario siempre usa una funcionalidad específica, enviarle ofertas relacionadas con esa funcionalidad.
        5. **`Eventos exclusivos en línea`**: Organizar eventos exclusivos como webinars o reuniones virtuales con expertos, donde solo los usuarios activos o pagos puedan participar.
        """],
    "Bajo": ["""
        1. **`Descuentos en renovación de suscripción`**: Ofrecer descuentos significativos si renuevan su suscripción o realizan pagos adicionales dentro de un corto periodo de tiempo
        2. **`Campañas de retargeting personalizado`**: Utilizar datos de comportamiento para ofrecerles promociones o contenidos personalizados que los inviten a retomar la actividad en la plataforma.
        3. **`Notificaciones personalizadas con ofertas de valor`**: Enviar recordatorios de productos o funciones que han utilizado previamente, junto con ofertas especiales (ejemplo: "Vuelve y consigue 10% de descuento en tu próxima compra").
        4. **`Descuentos por uso frecuente`**: Ofrecer descuentos o recompensas para aquellos usuarios que incrementen su actividad durante el mes (por ejemplo, si usan la plataforma 10 días consecutivos, obtienen un descuento del 15%).
        5. **`Recompensas por interacción con nuevas funciones`**: Incentivar a los usuarios a explorar nuevas características de la plataforma ofreciendo un beneficio como un mes adicional de suscripción o puntos de recompensa.
            """],

    "Medio": ["""
        1. **`Ofertas de reactivación personalizadas`**: Enviar un correo o notificación push ofreciendo un descuento importante o acceso a contenido exclusivo si regresan a la plataforma dentro de un plazo determinado.
        2. **`Recordatorio de funcionalidades no utilizadas`**: Utilizar los datos de comportamiento para enviar mensajes recordando las funcionalidades que no han sido exploradas por el usuario, ofreciendo tutoriales o guías rápidas.
        3. **`Campañas de contenido exclusivo para inactivos`**: Crear un catálogo de contenido exclusivo (tutoriales, seminarios web, o artículos premium) disponible solo para aquellos usuarios que regresen después de un periodo de inactividad.
        4. **`Ofrecer acceso a nuevas funcionalidades por tiempo limitado`**: Probar nuevas características de la plataforma de forma gratuita por un tiempo limitado a usuarios que han estado inactivos durante cierto periodo.
        5. **`Notificaciones de "última oportunidad"`**: Enviar un correo con un asunto como “Última oportunidad para obtener tus beneficios exclusivos”, creando un sentido de urgencia.
            """],

    "Alto": ["""
        1. **`Descuentos en el primer pago`**: Ofrecer descuentos agresivos o promociones de "primer pago gratis" si el usuario completa la conversión de gratuito a pago (por ejemplo, "Obtén un mes gratis si te suscribes ahora").
        2. **`Llamadas de atención personalizadas`**: Contactar directamente con estos usuarios a través de soporte al cliente o ventas para entender las razones de su baja actividad y ofrecer una solución personalizada (por ejemplo, “¿Te gustaría una sesión de asesoramiento para mejorar tu experiencia?”).
        3. **`Oferta de planes flexibles o a medida`**: Crear opciones de pago más flexibles o planes personalizados según el uso que hacen los usuarios. Ofrecer un “plan básico” para que comiencen a pagar a bajo costo.
        4. **`Campañas de reactivación urgente`**: Ofrecer grandes descuentos (como un 70% de descuento por tres meses) o beneficios adicionales si reactivan su cuenta dentro de las próximas 24 horas.
        5. **`Ofrecer sesiones de soporte o consultoría gratuita`**: Ofrecer sesiones gratuitas con un experto para guiar a los usuarios sobre cómo sacar el máximo provecho de la plataforma.
            """],

    "Muy Alto": ["""
        1. **`Campañas de recuperación con descuentos masivos`**: Ofrecer un descuento profundo como "90% de descuento en el primer mes si te suscribes ahora", para atraerlos a volver, aunque solo sea para probar la plataforma nuevamente.
        2. **`Encuestas de salida con incentivos`**: Enviar encuestas de salida con una recompensa por completarlas (por ejemplo, “dinos por qué te vas y recibe un 50% de descuento en tu próxima compra”).
        3. **`Planes gratuitos por tiempo limitado`**: Ofrecer un acceso completo y gratuito por 1 mes a todos los servicios premium, con la intención de engancharlos nuevamente a la plataforma.
        4. **`Comunicación directa de recuperación (SMS o Llamada)`**: Si es posible, contactar directamente con el usuario por teléfono o SMS para entender por qué no se están comprometiendo y ofrecer una oferta personalizada.
        5. **`Experiencia de onboarding personalizada`**: Crear una experiencia de reactivación guiada, con contenido paso a paso para que el usuario vuelva a usar la plataforma, mostrando cómo resolver sus puntos de dolor de manera efectiva.
        """]
}

def mostrar_estrategias(nivel_riesgo):
    estrategias = {
        "Muy Bajo": """
            1. **`Programa de recompensas por uso continuo`**: Implementar un sistema de puntos para los usuarios frecuentes, que puedan canjear por descuentos, contenido exclusivo o productos premium.
            2. **`Acceso anticipado a nuevas funcionalidades`**: Los usuarios más activos y pagos pueden ser invitados a probar nuevas funciones antes que el resto de los usuarios. Esto crea un sentido de exclusividad.
            3. **`Beneficios por referencia`**: Ofrecer recompensas por recomendar la plataforma a amigos o colegas. Esto podría ser un mes gratis o un descuento para ambos (referente y referido).
            4. **`Ofertas personalizadas para el perfil de uso`**: Ofrecer descuentos o beneficios exclusivos basados en el comportamiento de uso del cliente. Ejemplo: si un usuario siempre usa una funcionalidad específica, enviarle ofertas relacionadas con esa funcionalidad.
            5. **`Eventos exclusivos en línea`**: Organizar eventos exclusivos como webinars o reuniones virtuales con expertos, donde solo los usuarios activos o pagos puedan participar.
        """,

        "Bajo": """
            1. **`Descuentos en renovación de suscripción`**: Ofrecer descuentos significativos si renuevan su suscripción o realizan pagos adicionales dentro de un corto periodo de tiempo
            2. **`Campañas de retargeting personalizado`**: Utilizar datos de comportamiento para ofrecerles promociones o contenidos personalizados que los inviten a retomar la actividad en la plataforma.
            3. **`Notificaciones personalizadas con ofertas de valor`**: Enviar recordatorios de productos o funciones que han utilizado previamente, junto con ofertas especiales (ejemplo: "Vuelve y consigue 10% de descuento en tu próxima compra").
            4. **`Descuentos por uso frecuente`**: Ofrecer descuentos o recompensas para aquellos usuarios que incrementen su actividad durante el mes (por ejemplo, si usan la plataforma 10 días consecutivos, obtienen un descuento del 15%).
            5. **`Recompensas por interacción con nuevas funciones`**: Incentivar a los usuarios a explorar nuevas características de la plataforma ofreciendo un beneficio como un mes adicional de suscripción o puntos de recompensa.

        """,
        "Medio": """
            1. **`Ofertas de reactivación personalizadas`**: Enviar un correo o notificación push ofreciendo un descuento importante o acceso a contenido exclusivo si regresan a la plataforma dentro de un plazo determinado.
            2. **`Recordatorio de funcionalidades no utilizadas`**: Utilizar los datos de comportamiento para enviar mensajes recordando las funcionalidades que no han sido exploradas por el usuario, ofreciendo tutoriales o guías rápidas.
            3. **`Campañas de contenido exclusivo para inactivos`**: Crear un catálogo de contenido exclusivo (tutoriales, seminarios web, o artículos premium) disponible solo para aquellos usuarios que regresen después de un periodo de inactividad.
            4. **`Ofrecer acceso a nuevas funcionalidades por tiempo limitado`**: Probar nuevas características de la plataforma de forma gratuita por un tiempo limitado a usuarios que han estado inactivos durante cierto periodo.
            5. **`Notificaciones de "última oportunidad"`**: Enviar un correo con un asunto como “Última oportunidad para obtener tus beneficios exclusivos”, creando un sentido de urgencia.
        """,
        "Alto": """
            1. **`Descuentos en el primer pago`**: Ofrecer descuentos agresivos o promociones de "primer pago gratis" si el usuario completa la conversión de gratuito a pago (por ejemplo, "Obtén un mes gratis si te suscribes ahora").
            2. **`Llamadas de atención personalizadas`**: Contactar directamente con estos usuarios a través de soporte al cliente o ventas para entender las razones de su baja actividad y ofrecer una solución personalizada (por ejemplo, “¿Te gustaría una sesión de asesoramiento para mejorar tu experiencia?”).
            3. **`Oferta de planes flexibles o a medida`**: Crear opciones de pago más flexibles o planes personalizados según el uso que hacen los usuarios. Ofrecer un “plan básico” para que comiencen a pagar a bajo costo.
            4. **`Campañas de reactivación urgente`**: Ofrecer grandes descuentos (como un 70% de descuento por tres meses) o beneficios adicionales si reactivan su cuenta dentro de las próximas 24 horas.
            5. **`Ofrecer sesiones de soporte o consultoría gratuita`**: Ofrecer sesiones gratuitas con un experto para guiar a los usuarios sobre cómo sacar el máximo provecho de la plataforma.
        """,
        "Muy Alto": """
            1. **`Campañas de recuperación con descuentos masivos`**: Ofrecer un descuento profundo como "90% de descuento en el primer mes si te suscribes ahora", para atraerlos a volver, aunque solo sea para probar la plataforma nuevamente.
            2. **`Encuestas de salida con incentivos`**: Enviar encuestas de salida con una recompensa por completarlas (por ejemplo, “dinos por qué te vas y recibe un 50% de descuento en tu próxima compra”).
            3. **`Planes gratuitos por tiempo limitado`**: Ofrecer un acceso completo y gratuito por 1 mes a todos los servicios premium, con la intención de engancharlos nuevamente a la plataforma.
            4. **`Comunicación directa de recuperación (SMS o Llamada)`**: Si es posible, contactar directamente con el usuario por teléfono o SMS para entender por qué no se están comprometiendo y ofrecer una oferta personalizada.
            5. **`Experiencia de onboarding personalizada`**: Crear una experiencia de reactivación guiada, con contenido paso a paso para que el usuario vuelva a usar la plataforma, mostrando cómo resolver sus puntos de dolor de manera efectiva.
        """
    }

    if nivel_riesgo in estrategias:
        with st.expander(f"Estrategias de fidelización para **{nivel_riesgo}**"):
            st.markdown(estrategias[nivel_riesgo])


def color_con_riesgo(probabilidad):
        # Determinar nivel de riesgo
        if probabilidad <= 0.2:
            nivel = "Muy Bajo"
        elif probabilidad <= 0.4:
            nivel = "Bajo"
        elif probabilidad <= 0.6:
            nivel = "Medio"
        elif probabilidad <= 0.8:
            nivel = "Alto"
        else:
            nivel = "Muy Alto"

        colores_riesgo = {
            "Muy Bajo": "#2ca02c",
            "Bajo": "#98df8a",
            "Medio": "#ffcc00",
            "Alto": "#ff7f0e",
            "Muy Alto": "#d62728"
        }
        color = colores_riesgo[nivel]
        return color, nivel

def preparar_df_importancias(response):
    """
    Esta función recibe un diccionario `response` que contiene la clave 
    'CaracterísticasImportantes'. Extrae las variables más importantes, las ordena por valor absoluto 
    y devuelve el top 10 de las variables con mayor importancia.

    Parameters:
    - response (dict): Diccionario con las características y sus importancias.
    
    Returns:
    - DataFrame con el top 10 de las variables más importantes.
    """
    
    # Verificar si la clave 'CaracterísticasImportantes' existe en la respuesta
    if "CaracterísticasImportantes" not in response:
        raise ValueError("La clave 'CaracterísticasImportantes' no se encuentra en la respuesta.")

    # Convertir a DataFrame
    df_importancias = pd.DataFrame(response["CaracterísticasImportantes"])

    # Filtrar columnas que contienen '_importance' en su nombre
    df_importance = df_importancias[[col for col in df_importancias.columns if "_importance" in col]]
    
    # Renombrar columnas para eliminar '_importance' del nombre
    df_importance.columns = [col.replace("_importance", "") for col in df_importance.columns]

    # Transponer el DataFrame y renombrar las columnas
    df_top = df_importance.T.reset_index()
    df_top.columns = ["Variable", "Valor"]
    
    # Ordenar el DataFrame por el valor en la columna 'Valor'
    df_top = df_top.sort_values(by="Valor", ascending=False)

    # Top 10 por valor absoluto
    df_top_filtered = df_top.reindex(df_top["Valor"].abs().sort_values(ascending=False).index).head(10)

    return df_top_filtered

def plot_abonado_importancias(df):
    # Gráfico
    colors = ['red' if v > 0 else 'green' for v in df["Valor"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["Variable"], df["Valor"], color=colors)
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1)
    ax.set_xlabel("Impacto en riesgo", color='white')
    ax.set_title("Variables más influyentes para este abonado", color='white')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.tight_layout()
    plt.yticks(rotation=0, color='white')
    ax.tick_params(axis='x', colors='white')

    for i, v in enumerate(df["Valor"]):
        ha = 'left' if v > 0 else 'right'
        xpos = v + (0.0005 if v > 0 else 0.02)
        ax.text(
            xpos, i, f"{v:.2f}", color='black', va='center', ha=ha,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )
    return fig

def generar_frase_resumen(df_top_filtered, nivel_riesgo):
    # Separar variables por signo
    positivas = df_top_filtered[df_top_filtered["Valor"] > 0].sort_values(by="Valor", ascending=False).head(3)  # Las 3 que aumentan el riesgo
    negativas = df_top_filtered[df_top_filtered["Valor"] <= 0].sort_values(by="Valor", ascending=True).head(3)  # Las 3 que disminuyen el riesgo
    
    # Función para generar lista de variables
    def listar_variables(variables):
        if len(variables) == 0:
            return "ninguna"
        elif len(variables) == 1:
            return variables[0]
        elif len(variables) == 2:
            return f"{variables[0]} y {variables[1]}"
        else:
            return ", ".join(variables[:-1]) + f" y {variables[-1]}"
    
    # Obtener listas de variables
    positivas_variables = listar_variables(positivas["Variable"].tolist())
    negativas_variables = listar_variables(negativas["Variable"].tolist())
    
    # Ajustar frase según las variables disponibles
    if positivas_variables == "ninguna" and negativas_variables == "ninguna":
        frase_resumen = f"No se identificaron variables que aumenten ni que disminuyan el riesgo de abandono. El riesgo global es {nivel_riesgo}."
    elif positivas_variables == "ninguna":
        frase_resumen = f"El riesgo de abandono de este abonado es reducido principalmente por {negativas_variables}, resultando en un riesgo global {nivel_riesgo}."
    elif negativas_variables == "ninguna":
        frase_resumen = f"El riesgo de abandono de este abonado aumenta principalmente por {positivas_variables}, resultando en un riesgo global {nivel_riesgo}."
    else:
        frase_resumen = f"El riesgo de abandono de este abonado aumenta principalmente por {positivas_variables}, mientras que {negativas_variables} ayudan a reducirlo, resultando en un riesgo global {nivel_riesgo}."
    
    return frase_resumen

# --- 3. Explicación del modelo ---
def generar_explicacion_contexto(df):
    # Separar variables por signo
    positivas = df[df["Valor"] > 0]
    negativas = df[df["Valor"] <= 0]

    # Crear dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔺 Aumentan el riesgo")
        for _, row in positivas.iterrows():
            var = row["Variable"]
            imp = round(row["Valor"], 3)
            st.markdown(f"**{var}**: {imp}")

    with col2:
        st.markdown("### 🔻 Disminuyen el riesgo")
        for _, row in negativas.iterrows():
            var = row["Variable"]
            imp = round(row["Valor"], 3)
            st.markdown(f"**{var}**: {imp}")