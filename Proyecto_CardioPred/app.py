import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="CardioPred Triaje", page_icon="🫀", layout="centered")

# --- CSS PARA ESTILO MÉDICO ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model_and_pipeline():
    # Buscamos el modelo en la carpeta models
    try:
        model = joblib.load('Proyecto_CardioPred/models/cardio_pred_model.pkl')
        return model
    except FileNotFoundError:
        return None

pipeline = load_model_and_pipeline()

# --- INTERFAZ PRINCIPAL ---
st.title("🫀 CardioPred: Sistema de Soporte Clínico")
st.markdown("### Herramienta de Triaje de Riesgo Cardiovascular")
st.info("Ingrese los parámetros clínicos del paciente. El sistema utilizará un modelo de Random Forest calibrado (Umbral 0.30) para detectar riesgo.")

if pipeline is None:
    st.error("❌ Error Crítico: No se encuentra el archivo 'Proyecto_CardioPred/models/cardio_pred_model.pkl'. Asegúrate de haberlo descargado del notebook y colocado en la carpeta correcta.")
    st.stop()

# --- FORMULARIO DE ENTRADA ---
with st.container():
    st.write("#### 📋 Datos del Paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad (años)", 20, 100, 50)
        sex_label = st.selectbox("Sexo Biológico", ["Masculino", "Femenino"])
        cp_label = st.selectbox("Tipo de Dolor Torácico", 
                              ["Asintomático", "Angina Típica", "Angina Atípica", "Dolor No Anginoso"])
        resting_bp = st.number_input("Presión Arterial en Reposo (mmHg)", 80, 200, 120, help="Presión Sistólica")
        cholesterol = st.number_input("Colesterol Sérico (mg/dL)", 100, 600, 200)

    with col2:
        fasting_bs = st.selectbox("Glucemia en Ayunas > 120 mg/dL?", ["No", "Sí"])
        resting_ecg = st.selectbox("ECG en Reposo", ["Normal", "Anomalía ST-T", "Hipertrofia Ventricular"])
        max_hr = st.number_input("Frecuencia Cardiaca Máxima", 60, 220, 150)
        ex_angina = st.selectbox("¿Angina inducida por ejercicio?", ["No", "Sí"])
        oldpeak = st.number_input("Depresión del ST (Oldpeak)", 0.0, 6.0, 0.0, step=0.1)
        slope_label = st.selectbox("Pendiente del Segmento ST", ["Ascendente (Up)", "Plana (Flat)", "Descendente (Down)"])

# --- MAPEO DE DATOS (HUMANO -> MODELO) ---
# Convertimos las selecciones de texto a los números que aprendió el modelo
input_dict = {
    'age': age,
    'sex': 1 if sex_label == 'Masculino' else 0,
    'chest_pain_type': 1 if cp_label == "Angina Típica" else (2 if cp_label == "Angina Atípica" else (3 if cp_label == "Dolor No Anginoso" else 4)),
    'resting_bp_s': resting_bp,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': 1 if fasting_bs == "Sí" else 0,
    'resting_ecg': 0 if resting_ecg == "Normal" else (1 if resting_ecg == "Anomalía ST-T" else 2),
    'max_heart_rate': max_hr,
    'exercise_angina': 1 if ex_angina == "Sí" else 0,
    'oldpeak': oldpeak,
    'ST_slope': 1 if slope_label == "Ascendente (Up)" else (2 if slope_label == "Plana (Flat)" else 3)
}

# Crear DataFrame (Esto es lo que entra al Pipeline)
input_df = pd.DataFrame([input_dict])

# --- BOTÓN DE PREDICCIÓN ---
st.markdown("---")
if st.button("🚨 ANALIZAR RIESGO AHORA"):
    
    # 1. Obtener probabilidad bruta
    # El pipeline se encarga de escalar y transformar internamente
    probability = pipeline.predict_proba(input_df)[0][1]
    
    # 2. Aplicar Umbral Clínico (0.30 como definimos en el notebook)
    THRESHOLD = 0.30
    prediction = 1 if probability >= THRESHOLD else 0
    
    # 3. Mostrar Resultados
    st.subheader("Resultados del Análisis")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediction == 1:
            st.markdown("### 🔴 RIESGO ALTO")
        else:
            st.markdown("### 🟢 RIESGO BAJO")
            
    with col_res2:
        st.progress(int(probability * 100))
        st.write(f"Probabilidad estimada por IA: **{probability:.1%}**")
        
        if prediction == 1:
            st.warning(f"⚠️ El modelo ha detectado patrones asociados a patología cardiaca (Probabilidad > {THRESHOLD*100}%). Se recomienda derivación prioritaria.")
        else:
            st.success("✅ Los patrones ingresados sugieren un perfil fisiológico estable. Mantener controles rutinarios.")

    # Expander para ver los datos crudos (Debugging / Transparencia)
    with st.expander("Ver datos técnicos procesados"):
        st.dataframe(input_df)