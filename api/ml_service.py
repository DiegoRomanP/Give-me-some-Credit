import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 1. Cargar Artefactos globalmente (se ejecuta una vez al levantar el servidor)
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/xgboost_final.json"

artifacts = joblib.load(PREPROCESSOR_PATH)
imputer = artifacts['imputer']
mediana_edad_referencia = artifacts['mediana_edad_referencia']

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

def preprocess_payload(payload_dict: dict) -> pd.DataFrame:
    """Aplica las transformaciones exactas al payload entrante."""
    df = pd.DataFrame([payload_dict])
    
    # 1. Tratar edad
    if df.loc[0, 'age'] == 0:
        df.loc[0, 'age'] = mediana_edad_referencia
        
    # 2. Tratar retrasos anómalos
    columnas_retrasos = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    for col in columnas_retrasos:
        if df.loc[0, col] > 90:
            df.loc[0, col] = np.nan
            
    # 3. Transformaciones Logarítmicas (Crucial antes del imputer)
    columnas_log = ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']
    for col in columnas_log:
        df[col] = np.log1p(df[col].astype(float))
        
    df['MonthlyIncome_Missing'] = 1 if pd.isna(df.loc[0, 'MonthlyIncome']) else 0
    
    # 4. Imputación MICE
    cols_impute = ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents']
    df_subset = df[cols_impute].astype('float64')
    imputed_array = imputer.transform(df_subset)
    
    df['MonthlyIncome'] = imputed_array[:, 0]
    df['NumberOfDependents'] = imputed_array[:, 3]
    
    # Asegurar orden de columnas esperado por XGBoost (debe coincidir con X_train)
    features_order = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
        'MonthlyIncome_Missing'
    ]
    return df[features_order]

def get_prediction(payload_dict: dict) -> dict:
    # Preprocesar
    df_processed = preprocess_payload(payload_dict)
    
    # Inferencia (obtenemos la probabilidad de la clase 1: impago)
    probability = float(model.predict_proba(df_processed)[0, 1])
    
    # Clasificación de negocio lógica (ajustable según el caso)
    if probability < 0.3:
        risk = "Bajo"
    elif probability < 0.7:
        risk = "Medio"
    else:
        risk = "Alto"
        
    return {
        "default_probability": round(probability, 4),
        "risk_level": risk
    }