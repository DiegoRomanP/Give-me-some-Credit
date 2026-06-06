FROM python:3.10-slim

WORKDIR /app

# Copiar e instalar requerimientos del frontend
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la interfaz de Streamlit
COPY frontend/ ./frontend

EXPOSE 8501

# Configuración óptima para correr Streamlit dentro de un contenedor sin telemetría ni prompts interactivos
ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]