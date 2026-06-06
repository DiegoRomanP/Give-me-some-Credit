import streamlit as st
import requests
import os
# Configuración básica de la página
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")

# Lee la variable de entorno, si no existe, usa localhost (para poder seguir probando sin Docker)
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")

st.title("📊 Credit Default Risk Dashboard")
st.markdown("Ingrese los datos financieros del cliente para evaluar su probabilidad de impago en los próximos 2 años.")

# Usamos un formulario para enviar todos los datos a la vez
with st.form("credit_form"):
    st.subheader("Datos del Prestatario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad (años)", min_value=18, max_value=120, value=35)
        MonthlyIncome = st.number_input("Ingreso Mensual ($)", min_value=0.0, value=5000.0, step=100.0)
        NumberOfDependents = st.number_input("Número de Dependientes", min_value=0, max_value=20, value=0)
        DebtRatio = st.number_input("Ratio de Endeudamiento", min_value=0.0, value=0.3, format="%.4f")
        RevolvingUtilizationOfUnsecuredLines = st.number_input("Utilización de Líneas de Crédito", min_value=0.0, value=0.5, format="%.4f")

    with col2:
        NumberOfOpenCreditLinesAndLoans = st.number_input("Préstamos y Líneas Abiertas", min_value=0, value=5)
        NumberRealEstateLoansOrLines = st.number_input("Préstamos Hipotecarios", min_value=0, value=1)
        st.markdown("**Historial de Retrasos**")
        Time30_59 = st.number_input("Retrasos 30-59 días", min_value=0, value=0)
        Time60_89 = st.number_input("Retrasos 60-89 días", min_value=0, value=0)
        Time90Late = st.number_input("Retrasos +90 días", min_value=0, value=0)

    # Botón de envío
    submitted = st.form_submit_button("Analizar Riesgo Crediticio")

if submitted:
    # Preparar el payload respetando los alias de Pydantic en FastAPI
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": Time30_59,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome if MonthlyIncome > 0 else None, # Enviar None si es 0 para probar MICE
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": Time90Late,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": Time60_89,
        "NumberOfDependents": NumberOfDependents
    }

    with st.spinner('Procesando inferencia mediante XGBoost...'):
        try:
            # Hacer la petición POST a la API
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                prob = data["default_probability"]
                risk_level = data["risk_level"]
                
                # Mostrar Resultados Visuales
                st.divider()
                st.subheader("Resultados del Análisis")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label="Probabilidad de Impago", value=f"{prob * 100:.2f}%")
                
                with res_col2:
                    if risk_level == "Bajo":
                        st.success(f"Nivel de Riesgo: {risk_level} ✅")
                    elif risk_level == "Medio":
                        st.warning(f"Nivel de Riesgo: {risk_level} ⚠️")
                    else:
                        st.error(f"Nivel de Riesgo: {risk_level} 🚨")
            else:
                st.error(f"Error de la API: Código {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con el backend. Asegúrate de que FastAPI esté corriendo en el puerto 8000.")