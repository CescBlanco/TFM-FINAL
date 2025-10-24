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

from funciones_streamlit import *

col1, colspace, col3 = st.columns([1,3,1])

with col1:
    # Mostrar el logo gym
    st.image(LOGO_GYM, width=175)

with col3:
    st.image(LOGO_AYUNTAMIENTO, width=175)
# Configuración de la página
st.set_page_config(page_title="App de Predicción de Abandono", layout="wide")
# Usar markdown para centrar el título
st.markdown("<h1 style='text-align: center; color: #66BB6A;'>Predicción de Abandono: CEM Horta Esportiva</h1>", unsafe_allow_html=True)



# Mostrar un breve saludo
#st.subheader("Bienvenido a la herramienta de predicción de abandono de abonados")


# Tabs para las diferentes opciones
tabs = st.tabs([":bar_chart: Datos inventados", ":id: Un abonado", ":memo: Múltiples abonados", ":mag: Valoración modelos"])

# ------------------- #
# TAB 1: Datos individuales
# ------------------- #
with tabs[0]:
    st.markdown("<h2 style='color: #888;'>📝 Datos de Entrada del Abonado</h2>", unsafe_allow_html=True)

    st.write("Por favor, ingresa los datos del abonado para realizar la predicción.")
    userdata = input_userdata()  # Suponiendo que esta función obtiene los datos del usuario
    # Estilo con borde y sombra
    # Sección de predicción: claramente diferenciada
    st.write('----')
    st.markdown("<h3 style='color: #888;'>🔮 Realizar predicción</h3>", unsafe_allow_html=True)
    st.write("Haz clic en el botón para realizar la predicción sobre el abandono del abonado.")

    if st.button("🚀 Iniciar Predicción para un abonado inventado", key="btn_individual"):
        st.write("Calculando la predicción del abandono... ")

        resultados = []
        # Convertir los valores booleanos a True/False, sin cambiar a 1/0
        for col in BOOL_COL:
            if col in userdata:
                userdata[col] = True if userdata[col] else False
        
        # Verificar que los datos del usuario sean completos
        required_columns = set(COLUMNAS_MODELO)
        if not required_columns.issubset(userdata.keys()):
            st.error("⚠️ Faltan algunas columnas necesarias.")
        else:
                # Realizar la predicción
            response = obtener_predicciones_api("predecir_abandono_socio_simulado/", userdata)
            st.success("✅ Predicción obtenida")
            st.balloons()  
            if isinstance(response, dict):  # Verifica que la respuesta es un diccionario
                res = response
                res['IdPersona'] = res.get('IdPersona', 'Simulado')  # Asignar un id simulado si no existe
                resultados.append(res)
            else:
                st.warning(f"❌ Error en la predicción para IdPersona simulado")

        # Mostrar los resultados en formato tabla
            df_resultados = pd.DataFrame(resultados)
            if not df_resultados.empty:
                st.write("Resultados de predicción:")
                df_resultados = df_resultados.rename(columns={
                    'ProbabilidadAbandono': 'Probabilidad de Abandono',
                    'NivelRiesgo': 'Nivel de Riesgo'
                })

                st.dataframe(df_resultados)
             # Verificar si la respuesta es una cadena JSON y convertirla a un diccionario si es necesario
                try:
                    if isinstance(response, str):  # Si es una cadena
                        response = json.loads(response)  # Convertir de JSON a diccionario
                    
                    
                
                except json.JSONDecodeError as e:
                    st.error(f"Error al parsear la respuesta JSON: {e}")
                    

                # Ahora podemos procesar la respuesta
                if isinstance(response, dict):  # Si la respuesta es un diccionario
                    id_persona = response.get("IdPersona")
                    nivel_riesgo = response.get("NivelRiesgo")
                    
                    st.write(f"Nivel de Riesgo para ID {id_persona}: {nivel_riesgo}")

                    # Mostrar las estrategias de fidelización correspondientes al nivel de riesgo
                    if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:
                        with st.expander(f"Estrategias de fidelización para el ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                            for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                                st.markdown(estrategia)
                        st.balloons()  
                    else:
                        st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")
                else:
                    st.error("La respuesta no es un diccionario válido.")
            else:
                st.error("⚠️ No se obtuvo respuesta de la API.")
                

# ------------------- #
# TAB 1: Un ID
# ------------------- #
with tabs[1]:

    st.markdown("<h2 style='color: #888;'>Predicción por un abonado</h2>", unsafe_allow_html=True)
    id_persona = st.number_input("Introduce el ID de la persona", min_value=0, step=1)

    st.write('----')
    st.markdown("<h3 style='color: #888;'>🔮 Realizar predicción</h3>", unsafe_allow_html=True)
    st.write("Haz clic en el botón para realizar la predicción sobre el abandono del abonado.") 
    import json


    if st.button("🚀 Iniciar Predicción por un abonado", key="btn_id"):
        st.write("Calculando la predicción del abandono... ")
        data = {"IdPersona": id_persona}
        response = obtener_predicciones_api("predecir_abandono_por_id/", data)

        if response:
            st.success("✅ Predicción obtenida")
           
            st.json(response)  # Mostrar la respuesta para verificar su contenido

            # Verificar si la respuesta es una cadena JSON y convertirla a un diccionario si es necesario
            try:
                if isinstance(response, str):  # Si es una cadena
                    response = json.loads(response)  # Convertir de JSON a diccionario
                              
            except json.JSONDecodeError as e:
                st.error(f"Error al parsear la respuesta JSON: {e}")
                

            # Ahora podemos procesar la respuesta
            if isinstance(response, dict):  # Si la respuesta es un diccionario
                id_persona = response.get("IdPersona")
                nivel_riesgo = response.get("NivelRiesgo")
                
                st.write(f"Nivel de Riesgo para ID {id_persona}: {nivel_riesgo}")

                # Mostrar las estrategias de fidelización correspondientes al nivel de riesgo
                if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:
                    with st.expander(f"Estrategias de fidelización para el abonado con ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                        for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                            st.markdown(estrategia)
                    st.balloons()  
                else:
                    st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")
            else:
                st.error("La respuesta no es un diccionario válido.")
        else:
            st.error("⚠️ No se obtuvo respuesta de la API.")

# ------------------- #
# TAB 2: Múltiples IDs
# ------------------- #
with tabs[2]:
    st.markdown("<h2 style='color: #888;'>Predicción por múltiples abonados</h2>", unsafe_allow_html=True)

    ids_input = st.text_area("Introduce los IDs de los abonados separados por comas", value="123,456,789")

    st.markdown("<h3 style='color: #888;'>🔮 Realizar predicción</h3>", unsafe_allow_html=True)

    st.write("Haz clic en el botón para realizar la predicción sobre el abandono del abonado.")

    

    if st.button("🚀 Iniciar Predicción por múltiples abonados", key="btn_ids"):
        st.write("Calculando la predicción del abandono... ")
        try:
            ids_list = [int(id_.strip()) for id_ in ids_input.split(",") if id_.strip()]
            data = {"Ids": ids_list}
            response = obtener_predicciones_api("predecir_abandono_por_ids/", data)

            if response:
                st.success("✅ Predicciones obtenidas")
               
                st.json(response)  # Mostrar la respuesta para su revisión

                # Verificar si la respuesta es una cadena JSON y convertirla a un diccionario si es necesario
                try:
                    if isinstance(response, str):  # Si la respuesta es una cadena
                        response = json.loads(response)  # Convertir de JSON a diccionario
                        st.write("Respuesta convertida a diccionario:")  # Mostrar respuesta convertida
                        st.json(response)
                except json.JSONDecodeError as e:
                    st.error(f"Error al parsear la respuesta JSON: {e}")

                # Procesar la respuesta para cada ID
                if isinstance(response, list):  # Si la respuesta es una lista
                    for prediccion in response:
                        # Asegurarse de que la predicción tenga la estructura correcta
                        id_persona = prediccion.get("IdPersona")
                        nivel_riesgo = prediccion.get("NivelRiesgo")
                        
                        st.write(f"Nivel de Riesgo para ID {id_persona}: {nivel_riesgo}")

                        # Mostrar las estrategias de fidelización correspondientes al nivel de riesgo
                        if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:
                            with st.expander(f"Estrategias de fidelización para el abonado con ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                                for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                                    st.markdown(estrategia)
                            st.balloons()  
                        else:
                            st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")
                else:
                    st.error("La respuesta no es una lista válida.")
            else:
                st.error("⚠️ No se obtuvo respuesta de la API.")

        except ValueError:
            st.error("⚠️ Por favor, introduce solo números separados por comas")



# ------------------- #
# TAB 3: Valoración
# ------------------- #
with tabs[3]: 
        df_archivo_global_exp3, df_archivo_persona_ex3, df_archivo_preds_ex3 = encontrar_csv_inferencias(NAME_EXPERIMENT_3, FOLDER_DESTINO_3, RUN_ID_INF_3)
        df_archivo_global_exp2, df_archivo_persona_ex2, df_archivo_preds_ex2 = encontrar_csv_inferencias(NAME_EXPERIMENT_2, FOLDER_DESTINO_3, RUN_ID_INF_2)
        df_archivo_global_exp1, df_archivo_persona_ex1, df_archivo_preds_ex1 = encontrar_csv_inferencias(NAME_EXPERIMENT_1, FOLDER_DESTINO_1, RUN_ID_INF_1)

        #Encontrar las metricas del modelo usado: Experimento 3 y su inferencia
        auc_exp3, accuracy_exp3, f1_exp3, recall_exp3= encontrar_metricas_experimento(NAME_EXPERIMENT_3, metric=METRIC)
        accuracy, auc, f1, recall= encontrar_metricas_inferencia(RUN_ID_INF_3)
        
        view_option = st.radio("Elige la vista:", ("Mostrar modelo entrenado", "Mostrar modelo post inferencia"), horizontal=True)

        if view_option == 'Mostrar modelo entrenado':
            # Ruta al archivo CSV en tu sistema local
            file_path_inicial = 'mlops_api\data_mlops_api\dataframe_final_abonado.csv'

            # Leer el CSV directamente
            df_modelo_inicial = pd.read_csv(file_path_inicial)
            
            
            st.markdown("<h3 style='color: #888;'>Justificación para experimento 1 (No usado):</h3>", unsafe_allow_html=True)
                     
            col1_exp1, col2_exp1 = st.columns(2)

            with col1_exp1:
                fig_exp_1= plots_experimentos_sinuso(df_modelo_inicial, 'TotalPagadoEconomia')
                st.pyplot(fig_exp_1)

            with col2_exp1:
                fig_importnacias_exp1= plot_importancias(df_archivo_global_exp1)
                # Mostrar gráfico en Streamlit
                st.pyplot(fig_importnacias_exp1)
            
            st.markdown(' ')
            st.markdown("""
                🧑‍💻📊 Interpretación técnica:
                        
                - A partir de un pago de 600€, la probabilidad de abandono es casi nula.
                - Esto sugiere que el modelo está aprendiendo un patrón determinista: si un usuario paga más de 600, se clasifica automáticamente como activo.
                - **`"TotalPagadoEconomía"`**  domina el modelo, dejando de lado otras variables relevantes.
                - Esta dominancia lleva a una clasificación sesgada y menos precisa, especialmente para usuarios que pagan menos pero cuyo comportamiento de abandono depende de otros factores.
                - **`Decisión`** : Se decide eliminar esta variable para evitar el sesgo y permitir que el modelo considere mejor otras variables.        
                         """)

            st.markdown("<h3 style='color: #888;'>Justificación para experimento 2 (No usado):</h3>", unsafe_allow_html=True)
        
            col1_exp2, col2_exp2 = st.columns(2)

            with col1_exp2:
                fig_exp_2= plots_experimentos_sinuso(df_modelo_inicial, 'VidaGymMeses')
                st.pyplot(fig_exp_2)
    
            with col2_exp2:
                fig_importnacias_exp2= plot_importancias(df_archivo_global_exp2)
                # Mostrar gráfico en Streamlit
                st.pyplot(fig_importnacias_exp2)

            st.markdown(' ')
            st.markdown("""
                🧑‍💻📊 Interpretación técnica:
                        
                - Los usuarios que abandonan tienden a tener menos meses de suscripción, mientras que los que permanecen activos tienen más tiempo en el gimnasio.
                - El modelo podría estar aprendiendo un patrón determinista: si un cliente tiene más de un valor específico de meses (aproximadamente 150 meses), se clasifica automáticamente como no abandono.
                - Este patrón podría hacer que el modelo se sobreajuste, ignorando otras variables importantes.
                - **`Decisión`**: Se decide prescindir de esta variable para evitar que el modelo dependa de este valor umbral y así mejorar la inclusión de otras características.        
                """)
            
            st.markdown("<h3 style='color: #888;'>Justificación para la elección del experimento 3:</h3>", unsafe_allow_html=True)       
            
            st.markdown(f"""
                Rendimiento del modelo:
                        
                - **`AUC`**: {auc_exp3} → Excelente capacidad de distinguir entre quienes se quedan y quienes abandonan.
                - **`Accuracy`**: {accuracy_exp3} → Modelo fiable en general.
                - **`F1-score`**: {f1_exp3} → Buen equilibrio entre evitar falsos positivos y capturar verdaderos abandonos.
                - **`Recall`**: {recall_exp3} → Detecta casi 8 de cada 10 abonados que realmente abandonarían.

                **`Valor de negocio`**: Permite dirigir campañas de retención de manera efectiva, priorizando a los abonados en riesgo.

                **`Comparativa`**: Este experimento supera a otros modelos porque maximiza la detección de abandonos sin generar demasiadas falsas alarmas.
                         """)
            # Visualización del gráfico
            st.markdown("<h3 style='color: #888;'>Visualización de la importancia de las variables para el modelo:</h3>", unsafe_allow_html=True)
            
            
            fig_importnacias_exp3= plot_importancias(df_archivo_global_exp3)
            # Mostrar gráfico en Streamlit
            st.pyplot(fig_importnacias_exp3)

           
            # Crear un expnder (spinner) para mostrar más información técnica y de negocio
            with st.expander("🔍 Más información sobre la importancia de variables"):
                # **Parte técnica - Data Science:**
                st.markdown("""🧑‍💻📊 Interpretación técnica (Data Scientist / Data Analyst):""")

                st.markdown("""
                - **`DiasActivo`**: La cantidad de días que un abonado ha estado activo es la **variable más importante**. Los abonados con menos días activos tienen una mayor probabilidad de abandonar.
                - **`TotalVisitas`**: La cantidad total de visitas realizadas por un abonado es otro factor crucial. Un abonado que visita con regularidad es menos probable que abandone.
                - **`Edad`**: La edad del abonado tiene una relación importante con el abandono. Diferentes rangos de edad podrían tener diferentes comportamientos en cuanto a su lealtad y retención.
                - **`VisitasPrimerTrimestre`**: Las visitas en el primer trimestre de la suscripción son un indicador clave. Un alto número de visitas en este periodo podría predecir una mayor retención a largo plazo.
                - **`VisitasUlt180` y `VisitasUlt90`**: Las visitas recientes (últimos 180 y 90 días) también son importantes. Si un abonado ha estado menos activo recientemente, es más probable que abandone.
                - **`TienePagos`**: El abonado que ha realizado pagos regularmente es menos probable que abandone. Esto podría indicar un mayor compromiso o satisfacción con el servicio.
                """)
                st.markdown("""💼📈 Interpretación práctica de negocio:""")
            
                # **Parte de negocio - Interpretación práctica:**
                st.markdown("""
                - **`DiasActivo`**: Un abonado con pocos días activos debería ser un objetivo prioritario para campañas de retención, ya que es más probable que abandone. Considera ofrecer incentivos para aumentar la actividad.
                - **`TotalVisitas`**: Los abonados con pocas visitas son más propensos a abandonar. Para ellos, una estrategia podría ser ofrecer promociones de visitas o recordatorios personalizados.
                - **`Edad`**: La edad es una variable importante a tener en cuenta. Dependiendo del grupo de edad, las preferencias y expectativas sobre el servicio pueden variar. Los jóvenes podrían necesitar una experiencia más dinámica, mientras que los abonados más mayores podrían estar más interesados en un enfoque centrado en el bienestar.
                - **`VisitasPrimerTrimestre`**: Aprovecha la oportunidad en los primeros tres meses para enganchar a los nuevos abonados con una buena experiencia. Esto aumentará las probabilidades de retención a largo plazo.
                - **`VisitasUlt180` y `VisitasUlt90`**: Los abonados que no han visitado recientemente podrían estar en riesgo de abandonar. Ofrecer promociones o actividades especiales para reactivar su participación puede ser una estrategia clave.
                - **`TienePagos`**: Los abonados que pagan regularmente son menos propensos a abandonar. Tal vez se podría utilizar esta información para premiar a los abonados más fieles con beneficios exclusivos.
                """)

            st.markdown("<h3 style='color: #888;'>📝 Resumen del visual:</h3>", unsafe_allow_html=True)       


            # Sección visual con texto resumido
            st.markdown(""" 
            - **Las variables más importantes para predecir el abandono son**:
                - **`DiasActivo`**: El tiempo de actividad del abonado es crucial. Los abonados con menos días activos tienen mayor probabilidad de abandonar.
                - **`TotalVisitas`**: Un abonado que asiste más al centro está menos propenso a abandonar.
                - **`Edaad`**: La edad también influye en la probabilidad de abandono. Diferentes rangos de edad podrían tener diferentes comportamientos.
                
                        
            - **Las variables con menor impacto son**:
                - **`Diversidad_servicios_extra`** y **`UsoServiciosExtra`**: Aunque los servicios extra tienen su valor, el comportamiento central (actividad y visitas) es más importante.
            
                        
            - **Conclusiones finales**:
                - **`Mayor compromiso = menor probabilidad de abandono`**: Mantener a los usuarios activos y reactivar a los inactivos es clave.
                - **`Pagos = fuerte predictor de retención`**: Incrementar la conversión a pagos y cuidar la experiencia de los usuarios pagos reduce significativamente el abandono.
                - **`Visitas frecuentes = retención`**: Incentivar la actividad continua es clave para mantener a los abonados.
                - **`Usuarios jóvenes = mayor riesgo`**: Necesitan estrategias de retención personalizadas.
                - **`Personalización estacional`**: Promociones según preferencias de estación pueden mejorar la retención, aunque su impacto es menor.
            """)

            st.markdown("<h3 style='color: #888;'> 🔑 Posible recomendación:</h3>", unsafe_allow_html=True)       

            st.markdown("""           
            - **Actúa sobre la actividad del abonado**: Aumentar la actividad y la frecuencia de visitas será clave para retener a los abonados.
            - **Segmenta por edad y actividad**: Crea estrategias personalizadas según el nivel de actividad y la edad para mejorar la retención.
            """)

        elif view_option == 'Mostrar modelo post inferencia':
            # Ruta al archivo CSV en tu sistema local
            file_path = 'mlops_api/data_mlops_api/df_validacion_Experimento_v3.csv'

            
            # Leer el CSV directamente
            df_validacion = pd.read_csv(file_path)
            st.markdown("<h2 style='color: #999;'>🧑‍💻📊 Interpretación de la inferencia</h2>", unsafe_allow_html=True)

            col1_inf, col2_inf= st.columns(2)

            with col1_inf:
                st.markdown("<h5 style='color: #888;'>Rendimiento de la validación del modelo:</h5>", unsafe_allow_html=True)

                st.markdown(f"""
                                                
                    - **`AUC`**: {auc} → Muy buena capacidad para diferenciar entre abonados que se quedarán y los que abandonarán.
                    - **`Accuracy`**: {accuracy} → Modelo fiable en general.
                    - **`F1-score`**: {f1} → Mantiene un buen equilibrio entre precisión y detección de abandonos.
                    - **`Recall`**: {recall} → Detecta casi 8 de cada 10 abonados que realmente abandonarían.Recall: 84% → Detecta más de 8 de cada 10 abonados que realmente abandonarían, mejorando la identificación de riesgo frente al entrenamiento.

                    **`Comparativa`**: El modelo mantiene un buen equilibrio entre identificar abandonos y evitar falsas alertas, demostrando robustez tras la validación del modelo elegido.
                            """)
            

            df_persona_exp3 = df_archivo_preds_ex3.merge(df_archivo_persona_ex3, on='IdPersona', how='left')
            # Unir ambos DataFrames utilizando la columna "IdPersona"
            df_final_persona = pd.merge(df_persona_exp3, df_validacion[['IdPersona', "DiasActivo", "TotalVisitas", 'Edad', "VisitasPrimerTrimestre", "VisitasUlt180",  "TienePagos", "VisitasUlt90",
                                                                   "VisitasUltimoTrimestre", "EstFav_otono", "EstFav_verano"]], on='IdPersona', how='left', suffixes=('', '_inicial'))
           
             # Contar los valores y reordenarlos según tu preferencia

            with col2_inf:

                st.markdown("<h5 style='color: #888;'>Proporción de clientes por rango de abandono:</h5>", unsafe_allow_html=True)
                
                fig_piechart= piechart_edad(df_final_persona)
                # Mostrar el  en Streamlit
                st.pyplot(fig_piechart)

            col1results,col2results= st.columns(2)
           
            grouped_activos_reset, grouped_abandonados_reset=  tabla_recuento_resultados(df_final_persona)
  
            with col1results: 
                # Mostrar las tablas separadas en Streamlit
                st.markdown("<h3 style='color: #888;'>Clientes Activos:</h3>", unsafe_allow_html=True)
                st.table(grouped_activos_reset)  # Mostrar tabla de clientes activos

            with col2results:
                st.markdown("<h3 style='color: #888;'>Clientes Abandonados:</h3>", unsafe_allow_html=True)
                st.table(grouped_abandonados_reset)  # Mostrar tabla de clientes abandonados
            
            st.markdown("<h3 style='color: #888;'>Factores que afectan la probabilidad de abandono:</h3>", unsafe_allow_html=True)

            
            df_final_persona= categorizacion_variables_importancia(df_final_persona)
            
            opciones = [
                "Probabilidad de Abandono por Grupos de Edad",
                "Probabilidad de Abandono por Grupos de Días Activos",
                "Probabilidad de Abandono por Grupos de Visitas Últimos 180 Días",
                "Probabilidad de Abandono por Visitas Primer Trimestre",
                "Probabilidad de Abandono por Estación Favorita Otoño",
                "Probabilidad de Abandono por si Tiene Pagos"
            ]

            eleccion = st.selectbox("Elige un gráfico para ver:", opciones)

            # Mostrar gráfico según la selección
            mostrar_grafico_y_descripcion(eleccion, df_final_persona)          
           
            
            # Título y descripción
            st.markdown("<h3 style='color: #888;'>Estrategias de Fidelización:</h3>", unsafe_allow_html=True)
            st.markdown("Selecciona el nivel de riesgo de abandono de los usuarios para ver las estrategias de fidelización recomendadas.")

            # Selector de riesgo
            nivel_riesgo = st.selectbox("Selecciona el Nivel de Riesgo:", 
                                        ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"])

            # Acordeones para cada nivel
            if nivel_riesgo == "Muy Bajo":
                with st.expander("Estrategias de fidelización para **Muy Bajo**"):
                    st.markdown("""
                        1. **`Programa de recompensas por uso continuo`**: Implementar un sistema de puntos para los usuarios frecuentes, que puedan canjear por descuentos, contenido exclusivo o productos premium.
                        2. **`Acceso anticipado a nuevas funcionalidades`**: Los usuarios más activos y pagos pueden ser invitados a probar nuevas funciones antes que el resto de los usuarios. Esto crea un sentido de exclusividad.
                        3. **`Beneficios por referencia`**: Ofrecer recompensas por recomendar la plataforma a amigos o colegas. Esto podría ser un mes gratis o un descuento para ambos (referente y referido).
                        4. **`Ofertas personalizadas para el perfil de uso`**: Ofrecer descuentos o beneficios exclusivos basados en el comportamiento de uso del cliente. Ejemplo: si un usuario siempre usa una funcionalidad específica, enviarle ofertas relacionadas con esa funcionalidad.
                        5. **`Eventos exclusivos en línea`**: Organizar eventos exclusivos como webinars o reuniones virtuales con expertos, donde solo los usuarios activos o pagos puedan participar.

                    """)

            elif nivel_riesgo == "Bajo":
                with st.expander("Estrategias de fidelización para **Bajo**"):
                    st.markdown("""
                        1. **`Descuentos en renovación de suscripción`**: Ofrecer descuentos significativos si renuevan su suscripción o realizan pagos adicionales dentro de un corto periodo de tiempo
                        2. **`Campañas de retargeting personalizado`**: Utilizar datos de comportamiento para ofrecerles promociones o contenidos personalizados que los inviten a retomar la actividad en la plataforma.
                        3. **`Notificaciones personalizadas con ofertas de valor`**: Enviar recordatorios de productos o funciones que han utilizado previamente, junto con ofertas especiales (ejemplo: "Vuelve y consigue 10% de descuento en tu próxima compra").
                        4. **`Descuentos por uso frecuente`**: Ofrecer descuentos o recompensas para aquellos usuarios que incrementen su actividad durante el mes (por ejemplo, si usan la plataforma 10 días consecutivos, obtienen un descuento del 15%).
                        5. **`Recompensas por interacción con nuevas funciones`**: Incentivar a los usuarios a explorar nuevas características de la plataforma ofreciendo un beneficio como un mes adicional de suscripción o puntos de recompensa.
                    """)

            elif nivel_riesgo == "Medio":
                with st.expander("Estrategias de fidelización para **Medio**"):
                    st.markdown("""
                        1. **`Ofertas de reactivación personalizadas`**: Enviar un correo o notificación push ofreciendo un descuento importante o acceso a contenido exclusivo si regresan a la plataforma dentro de un plazo determinado.
                        2. **`Recordatorio de funcionalidades no utilizadas`**: Utilizar los datos de comportamiento para enviar mensajes recordando las funcionalidades que no han sido exploradas por el usuario, ofreciendo tutoriales o guías rápidas.
                        3. **`Campañas de contenido exclusivo para inactivos`**: Crear un catálogo de contenido exclusivo (tutoriales, seminarios web, o artículos premium) disponible solo para aquellos usuarios que regresen después de un periodo de inactividad.
                        4. **`Ofrecer acceso a nuevas funcionalidades por tiempo limitado`**: Probar nuevas características de la plataforma de forma gratuita por un tiempo limitado a usuarios que han estado inactivos durante cierto periodo.
                        5. **`Notificaciones de "última oportunidad"`**: Enviar un correo con un asunto como “Última oportunidad para obtener tus beneficios exclusivos”, creando un sentido de urgencia.
                    """)

            elif nivel_riesgo == "Alto":
                with st.expander("Estrategias de fidelización para **Alto**"):
                    st.markdown("""
                        1. **`Descuentos en el primer pago`**: Ofrecer descuentos agresivos o promociones de "primer pago gratis" si el usuario completa la conversión de gratuito a pago (por ejemplo, "Obtén un mes gratis si te suscribes ahora").
                        2. **`Llamadas de atención personalizadas`**: Contactar directamente con estos usuarios a través de soporte al cliente o ventas para entender las razones de su baja actividad y ofrecer una solución personalizada (por ejemplo, “¿Te gustaría una sesión de asesoramiento para mejorar tu experiencia?”).
                        3. **`Oferta de planes flexibles o a medida`**: Crear opciones de pago más flexibles o planes personalizados según el uso que hacen los usuarios. Ofrecer un “plan básico” para que comiencen a pagar a bajo costo.
                        4. **`Campañas de reactivación urgente`**: Ofrecer grandes descuentos (como un 70% de descuento por tres meses) o beneficios adicionales si reactivan su cuenta dentro de las próximas 24 horas.
                        5. **`Ofrecer sesiones de soporte o consultoría gratuita`**: Ofrecer sesiones gratuitas con un experto para guiar a los usuarios sobre cómo sacar el máximo provecho de la plataforma.
                    """)

            elif nivel_riesgo == "Muy Alto":
                with st.expander("Estrategias de Fidelización para **Muy Alto**"):
                    st.markdown("""
                        1. **`Campañas de recuperación con descuentos masivos`**: Ofrecer un descuento profundo como "90% de descuento en el primer mes si te suscribes ahora", para atraerlos a volver, aunque solo sea para probar la plataforma nuevamente.
                        2. **`Encuestas de salida con incentivos`**: Enviar encuestas de salida con una recompensa por completarlas (por ejemplo, “dinos por qué te vas y recibe un 50% de descuento en tu próxima compra”).
                        3. **`Planes gratuitos por tiempo limitado`**: Ofrecer un acceso completo y gratuito por 1 mes a todos los servicios premium, con la intención de engancharlos nuevamente a la plataforma.
                        4. **`Comunicación directa de recuperación (SMS o Llamada)`**: Si es posible, contactar directamente con el usuario por teléfono o SMS para entender por qué no se están comprometiendo y ofrecer una oferta personalizada.
                        5. **`Experiencia de onboarding personalizada`**: Crear una experiencia de reactivación guiada, con contenido paso a paso para que el usuario vuelva a usar la plataforma, mostrando cómo resolver sus puntos de dolor de manera efectiva.
                    """)


st.markdown("""
<footer style='text-align:center; font-size:12px; color:#888;'>
<p>© 2025 Cesc Blanco | Contacto: cesc.blanco98@gmail.com | Sígueme en LinkedIn: Cesc Blanco Arnau</p>
</footer>
""", unsafe_allow_html=True)

    # st.download_button(
    # label="Descargar CSV",
    # data=archivo,
    # file_name="archivo.csv",
    # mime="text/csv")