import streamlit as st
import pandas as pd
import requests
import json
import time
# URL de tu API local
API_URL = "http://localhost:8000"

st.set_page_config(page_title="App de Predicción de Abandono", layout="wide")
st.title("📊 App de Predicción de Abandono")

# Tabs para las diferentes opciones
tabs = st.tabs(["Datos inventados", "Múltiples datos", "Un ID", "Múltiples IDs"])

# Función para formulario individual
def input_userdata():
    Edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
    Sexo_Mujer = st.checkbox("Sexo Mujer")
    UsoServiciosExtra = st.checkbox("Uso Servicios Extra")
    ratio_cantidad_2025_2024 = st.number_input("Ratio cantidad 2025/2024", value=1.0, format="%.3f")
    Diversidad_servicios_extra = st.number_input("Diversidad servicios extra", min_value=0, max_value=100, value=1)
    TienePagos = st.checkbox("Tiene Pagos")
    TotalVisitas = st.number_input("Total Visitas", min_value=0, value=0)
    DiasActivo = st.number_input("Días Activo", min_value=0, value=0)
    VisitasUlt90 = st.number_input("Visitas Últimos 90 días", min_value=0, value=0)
    VisitasUlt180 = st.number_input("Visitas Últimos 180 días", min_value=0, value=0)
    TieneAccesos = st.checkbox("Tiene Accesos")
    VisitasPrimerTrimestre = st.number_input("Visitas Primer Trimestre", min_value=0, value=0)
    VisitasUltimoTrimestre = st.number_input("Visitas Último Trimestre", min_value=0, value=0)
    DiaFav_domingo = st.checkbox("Día Favorito Domingo")
    DiaFav_jueves = st.checkbox("Día Favorito Jueves")
    DiaFav_lunes = st.checkbox("Día Favorito Lunes")
    DiaFav_martes = st.checkbox("Día Favorito Martes")
    DiaFav_miercoles = st.checkbox("Día Favorito Miércoles")
    DiaFav_sabado = st.checkbox("Día Favorito Sábado")
    DiaFav_viernes = st.checkbox("Día Favorito Viernes")
    EstFav_invierno = st.checkbox("Estación Favorita Invierno")
    EstFav_otono = st.checkbox("Estación Favorita Otoño")
    EstFav_primavera = st.checkbox("Estación Favorita Primavera")
    EstFav_verano = st.checkbox("Estación Favorita Verano")

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
        "EstFav_verano": EstFav_verano
    }

# --------------------- #
# TAB 1: Datos individuales
# --------------------- #
with tabs[0]:
    st.subheader("Predicción por datos individuales")
    userdata = input_userdata()
    if st.button("Predecir", key="btn_individual"):
        response = requests.post(f"{API_URL}/predecir/", json=userdata)
        if response.status_code == 200:
            st.success("✅ Predicción obtenida")
            st.json(response.json())
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

# --------------------- #
# TAB 2: Múltiples datos (CSV)
# --------------------- #
with tabs[1]:
    st.header("Predicción por múltiples datos de abonados reales")
    API_URL = "http://localhost:8000/predecir/"
    columnas_modelo = [
        "Edad", "Sexo_Mujer", "UsoServiciosExtra", "ratio_cantidad_2025_2024",
        "Diversidad_servicios_extra", "TienePagos", "TotalVisitas", "DiasActivo",
        "VisitasUlt90", "VisitasUlt180", "TieneAccesos", "VisitasPrimerTrimestre",
        "VisitasUltimoTrimestre", "DiaFav_domingo", "DiaFav_jueves", "DiaFav_lunes",
        "DiaFav_martes", "DiaFav_miercoles", "DiaFav_sabado", "DiaFav_viernes",
        "EstFav_invierno", "EstFav_otono", "EstFav_primavera", "EstFav_verano"
    ]

    uploaded_file = st.file_uploader("Sube un archivo CSV con los datos")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Archivo cargado con {len(df)} filas")

        st.markdown("### Opciones de predicción")
        
        col1, col2 = st.columns(2)

        with col1:
            all_rows = st.checkbox("✅ Predecir todas las filas")

            if all_rows and st.button("🔮 Obtener predicciones (Todas)"):
                df_seleccion = df.copy()
                resultados = []

                bool_cols = [
                    "Sexo_Mujer", "UsoServiciosExtra", "TienePagos", "TieneAccesos",
                    "DiaFav_domingo", "DiaFav_jueves", "DiaFav_lunes", "DiaFav_martes",
                    "DiaFav_miercoles", "DiaFav_sabado", "DiaFav_viernes",
                    "EstFav_invierno", "EstFav_otono", "EstFav_primavera", "EstFav_verano"
                ]
                progreso = st.empty()  # Placeholder para mostrar progreso
                inicio = time.time()   # Tiempo inicial


                for idx, row in df_seleccion.iterrows():

                    fila_actual = idx + 1
                    total_filas = len(df_seleccion)


                    datos = row[columnas_modelo].to_dict()
                    for col in bool_cols:
                        if col in datos:
                            datos[col] = bool(datos[col])
                                # Mostrar progreso antes de hacer la petición
                    
                    tiempo_actual = time.time() - inicio
                    progreso.info(f"Procesando fila {fila_actual}/{total_filas} ⏱️ Tiempo: {tiempo_actual:.2f}s")

                    response = requests.post(API_URL, json=datos)

                    if response.status_code == 200:
                        res = response.json()
                        # Agregar campos para mostrar luego
                        res['IdPersona'] = row['IdPersona']
                        res['Edad'] = row['Edad']
                        res['Sexo_Mujer'] = datos['Sexo_Mujer']  # lo añadimos para convertirlo luego
                        resultados.append(res)
                        st.success("✅ Predicción obtenida y completadas en {duracion_total:.2f} segundos")

                    else:
                        resultados.append({"error": f"Status {response.status_code}", "IdPersona": row['IdPersona']})

                fin = time.time()
                duracion_total = fin - inicio

                # Ahora convertir Sexo_Mujer a texto y quitar campo booleano
                for res in resultados:
                    if 'Sexo_Mujer' in res:
                        res['Sexo'] = "Mujer" if res['Sexo_Mujer'] else "Hombre"
                        del res['Sexo_Mujer']


        with col2:
            num_rows_option = st.checkbox("🎯 Seleccionar número de filas")
            if num_rows_option:
                num_rows = st.number_input("Número de filas a predecir", min_value=1, max_value=len(df), value=5)
                if st.button("🔮 Obtener predicciones (N filas)"):
                    df_seleccion = df.head(num_rows)
                    resultados = []

                    bool_cols = [
                        "Sexo_Mujer", "UsoServiciosExtra", "TienePagos", "TieneAccesos",
                        "DiaFav_domingo", "DiaFav_jueves", "DiaFav_lunes", "DiaFav_martes",
                        "DiaFav_miercoles", "DiaFav_sabado", "DiaFav_viernes",
                        "EstFav_invierno", "EstFav_otono", "EstFav_primavera", "EstFav_verano"
                    ]
                    
                    progreso = st.empty()  # Placeholder para mostrar progreso
                    inicio = time.time()   # Tiempo inicial


                    for idx, row in df_seleccion.iterrows():

                        fila_actual = idx + 1
                        total_filas = len(df_seleccion)

                        datos = row[columnas_modelo].to_dict()
                        for col in bool_cols:
                            if col in datos:
                                datos[col] = bool(datos[col])
                        
                        tiempo_actual = time.time() - inicio
                        progreso.info(f"Procesando fila {fila_actual}/{total_filas} ⏱️ Tiempo: {tiempo_actual:.2f}s")
                        
                        response = requests.post(API_URL, json=datos)

                        if response.status_code == 200:
                            res = response.json()
                            res['IdPersona'] = row['IdPersona']
                            res['Edad'] = row['Edad']
                            res['Sexo_Mujer'] = datos['Sexo_Mujer']
                            resultados.append(res)
                        
                        else:
                            resultados.append({"error": f"Status {response.status_code}", "IdPersona": row['IdPersona']})

                    for res in resultados:
                        if 'Sexo_Mujer' in res:
                            res['Sexo'] = "Mujer" if res['Sexo_Mujer'] else "Hombre"
                            del res['Sexo_Mujer']
                    
                    fin = time.time()
                    duracion_total = fin - inicio

                    st.success(f"✅ Predicción obtenida y completadas en {duracion_total:.2f} segundos")

        df_resultados = pd.DataFrame(resultados)
        st.write("Resultados de predicción:")
        columnas_ordenadas = ["IdPersona", "Edad", "Sexo", "ProbabilidadAbandono", "NivelRiesgo"]
        df_resultados = df_resultados[columnas_ordenadas]
        df_resultados= df_resultados.rename(columns= {'ProbabilidadAbandono': 'Probabilidad de Abandono', 'NivelRiesgo': 'Nivel de Riesgo'})

        st.table(df_resultados.reset_index(drop=True))

 


# --------------------- #
# TAB 3: Un ID
# --------------------- #
with tabs[2]:
    st.subheader("Predicción por un ID")
    id_persona = st.number_input("Introduce el ID de la persona", min_value=0, step=1)

    if st.button("Predecir por ID", key="btn_id"):
        data = {"IdPersona": id_persona}
        response = requests.post(f"{API_URL}/predecir_por_id/", json=data)

        if response.status_code == 200:
            st.success("✅ Predicción obtenida")
            st.json(response.json())
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

# --------------------- #
# TAB 4: Múltiples IDs
# --------------------- #
with tabs[3]:
    st.subheader("Predicción por múltiples IDs")
    ids_input = st.text_area("Introduce IDs separados por comas", value="123,456,789")

    if st.button("Predecir por múltiples IDs", key="btn_ids"):
        try:
            ids_list = [int(id_.strip()) for id_ in ids_input.split(",") if id_.strip()]
            data = {"Ids": ids_list}
            response = requests.post(f"{API_URL}/predecir_por_ids/", json=data)

            if response.status_code == 200:
                st.success("✅ Predicciones obtenidas")
                st.json(response.json())
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except ValueError:
            st.error("⚠️ Por favor, introduce solo números separados por comas")
