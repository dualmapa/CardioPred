
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="CardioPred Triaje", page_icon="🫀", layout="centered")

# --- CSS PARA ESTILO MÉDICO (ojo: etiquetas HTML reales) ---
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

# --- CARGA DEL MODELO (robusta y con log de ruta) ---
@st.cache_resource
def load_model_and_pipeline():
    base = Path(__file__).parent

    # Candidatos más comunes
    candidates = [
        base / 'models' / 'cardio_pred_model.pkl',
        base / 'Proyecto_CardioPred' / 'models' / 'cardio_pred_model.pkl',
    ]

    # 1) Busca en candidatos conocidos
    for p in candidates:
        if p.exists():
            try:
                model = joblib.load(p)
                return model, str(p)
            except Exception as e:
                # Propaga el error real para verlo en Streamlit (Logs)
                raise RuntimeError(f"Fallo cargando el modelo desde {p}: {e}")

    # 2) Búsqueda recursiva en todo el repo (por si la estructura cambió)
    for p in base.rglob('cardio_pred_model.pkl'):
        try:
            model = joblib.load(p)
            return model, str(p)
        except Exception as e:
            raise RuntimeError(f"Fallo cargando el modelo desde {p}: {e}")

    # 3) No encontrado
    return None, None

pipeline, model_path = load_model_and_pipeline()

# --- INTERFAZ PRINCIPAL ---
st.title("🫀 CardioPred: Sistema de Soporte Clínico")
st.markdown("### Herramienta de Triaje de Riesgo Cardiovascular")
st.info("Ingrese los parámetros clínicos del paciente. El sistema utiliza un modelo de Random Forest calibrado (Umbral 0.30) para detectar riesgo.")

if pipeline is None:
    st.error("❌ Error Crítico: No se encontró el modelo en las rutas esperadas.\n"
             "Pruebe colocar el archivo en:\n"
             "- `models/cardio_pred_model.pkl`\n"
             "- `Proyecto_CardioPred/models/cardio_pred_model.pkl`\n"
             "Luego haga *Rerun* en Streamlit Cloud.")
    st.stop()
else:
    st.caption(f"🗂️ Modelo cargado desde: `{model_path}`")

# --- FORMULARIO DE ENTRADA ---
with st.container():
    st.write("#### 📋 Datos del Paciente")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (años)", min_value=20, max_value=100, value=50)
        sex_label = st.selectbox("Sexo Biológico", ["Masculino", "Femenino"])
        cp_label = st.selectbox("Tipo de Dolor Torácico",
                                ["Asintomático", "Angina Típica", "Angina Atípica", "Dolor No Anginoso"])
        resting_bp = st.number_input("Presión Arterial en Reposo (mmHg)", min_value=80, max_value=200, value=120,
                                     help="Presión Sistólica")
        cholesterol = st.number_input("Colesterol Sérico (mg/dL)", min_value=100, max_value=600, value=200)

    with col2:
        fasting_bs = st.selectbox("Glucemia en Ayunas > 120 mg/dL?", ["No", "Sí"])
        resting_ecg = st.selectbox("ECG en Reposo", ["Normal", "Anomalía ST-T", "Hipertrofia Ventricular"])
        max_hr = st.number_input("Frecuencia Cardiaca Máxima", min_value=60, max_value=220, value=150)
        ex_angina = st.selectbox("¿Angina inducida por ejercicio?", ["No", "Sí"])
        oldpeak = st.number_input("Depresión del ST (Oldpeak)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        slope_label = st.selectbox("Pendiente del Segmento ST", ["Ascendente (Up)", "Plana (Flat)", "Descendente (Down)"])

# --- MAPEO DE DATOS (HUMANO -> MODELO) ---
input_dict = {
    'age': age,
    'sex': 1 if sex_label == 'Masculino' else 0,
    'chest_pain_type': (
        1 if cp_label == "Angina Típica" else
        (2 if cp_label == "Angina Atípica" else
         (3 if cp_label == "Dolor No Anginoso" else 4))  # 4 = Asintomático
    ),
    'resting_bp_s': resting_bp,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': 1 if fasting_bs == "Sí" else 0,
    'resting_ecg': (0 if resting_ecg == "Normal" else
                    (1 if resting_ecg == "Anomalía ST-T" else 2)),
    'max_heart_rate': max_hr,
    'exercise_angina': 1 if ex_angina == "Sí" else 0,
    'oldpeak': oldpeak,
    'ST_slope': (1 if slope_label == "Ascendente (Up)" else
                 (2 if slope_label == "Plana (Flat)" else 3))
}
input_df = pd.DataFrame([input_dict])

# --- BOTÓN DE PREDICCIÓN ---
st.markdown("---")
if st.button("🚨 ANALIZAR RIESGO AHORA"):
    try:
        # 1. Probabilidad bruta
        if hasattr(pipeline, "predict_proba"):
            probability = float(pipeline.predict_proba(input_df)[:, 1][0])
        else:
            # Fallback muy básico (poco común con RF)
            probability = float(pipeline.predict(input_df))

        # 2. Umbral clínico
        THRESHOLD = 0.30
        prediction = 1 if probability >= THRESHOLD else 0

        # 3. Mostrar resultados
        st.subheader("Resultados del Análisis")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.markdown("### 🔴 RIESGO ALTO" if prediction == 1 else "### 🟢 RIESGO BAJO")
        with col_res2:
            st.progress(int(probability * 100))
            st.write(f"Probabilidad estimada por IA: **{probability:.1%}**")
            if prediction == 1:
                st.warning(f"⚠️ El modelo ha detectado patrones asociados a patología cardiaca "
                           f"(Probabilidad > {THRESHOLD*100}%). Se recomienda derivación prioritaria.")
            else:
                st.success("✅ Los patrones ingresados sugieren un perfil fisiológico estable. Mantener controles rutinarios.")

        with st.expander("Ver datos técnicos procesados"):
            st.dataframe(input_df)

    except Exception as e:
        st.error("❌ Ocurrió un error ejecutando la predicción. Revisa los *logs* de la app en Streamlit Cloud.")
        st.exception(e)  # Esto muestra el stack trace en la interfaz
