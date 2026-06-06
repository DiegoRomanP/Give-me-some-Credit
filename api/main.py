from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CreditPredictionRequest, CreditPredictionResponse
from api.ml_service import get_prediction

app = FastAPI(
    title="Risk Scoring API",
    description="API para predicción de riesgo crediticio usando XGBoost",
    version="1.0.0"
)

# Configuración CORS para permitir peticiones desde Streamlit u otros frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, reemplazar con la URL de Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API funcionando correctamente"}

@app.post("/predict", response_model=CreditPredictionResponse)
def predict_credit_risk(request: CreditPredictionRequest):
    try:
        # Extraer los datos del request respetando los alias originales (ej. los guiones)
        payload = request.dict(by_alias=True)
        
        # Obtener predicción del servicio ML
        result = get_prediction(payload)
        
        return result
    except Exception as e:
        # Registrar el error y devolver un 500 elegante
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(e)}")