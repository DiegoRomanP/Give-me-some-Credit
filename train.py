import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os

# Imports para imputación por pliegue en validación cruzada
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# ── DATOS ──
# Datos limpios (ya imputados) para evaluación inicial y modelo final
df_train_clean = pd.read_csv('data/train_clean.csv')
df_test_clean = pd.read_csv('data/test_clean.csv')

# Datos pre-imputación (con NaNs) para validación cruzada sin fuga de datos
df_train_pre = pd.read_csv('data/train_pre_impute.csv')

df_train_clean = df_train_clean.dropna(subset=['SeriousDlqin2yrs'])
df_train_pre = df_train_pre.dropna(subset=['SeriousDlqin2yrs'])

# Separar Features (X) y Target (y) del TRAIN LIMPIO (para eval inicial y modelo final)
X = df_train_clean.drop(columns=['SeriousDlqin2yrs', 'Id'])
y = df_train_clean['SeriousDlqin2yrs']

# Features y Target PRE-IMPUTACIÓN (para CV — la imputación ocurrirá por pliegue)
X_cv = df_train_pre.drop(columns=['SeriousDlqin2yrs', 'Id'])
y_cv = df_train_pre['SeriousDlqin2yrs']

# Columnas que requieren imputación (mismo orden que en preprocess.py)
cols_impute = ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents']

# Hacemos el split local para evaluación
X_train_local, X_val_local, y_train_local, y_val_local = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calcular pesos de clases para el desbalance
peso_balanceo = y_train_local.value_counts()[0] / y_train_local.value_counts()[1]

# Configurar el modelo XGBoost
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=peso_balanceo,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc'
)

# Entrenar modelo
xgb_model.fit(X_train_local, y_train_local)

# EVALUACIÓN DE PRECISIÓN LOCAL
y_val_pred = xgb_model.predict(X_val_local)
y_val_proba = xgb_model.predict_proba(X_val_local)[:, 1]

print("=== REPORTE DE CLASIFICACIÓN (EVALUACIÓN LOCAL) ===")
print(classification_report(y_val_local, y_val_pred, target_names=['Buen Pagador (0)', 'Impago (1)']))
print(f"ROC-AUC Local: {roc_auc_score(y_val_local, y_val_proba):.4f}")



# Reentrenar con TODO el dataset limpio de entrenamiento para maximizar el aprendizaje
print("\nReentrenando con el 100% de los datos de Train...")
xgb_model.fit(X, y)

# Predecir sobre el df_test_clean
# Nos aseguramos de eliminar el Id si lo tiene, para que las columnas coincidan
X_test_final = df_test_clean.drop(columns=['Id', 'SeriousDlqin2yrs'], errors='ignore')

# Kaggle evalúa en base a las PROBABILIDADES de impago, no la clase absoluta
probabilidades_test = xgb_model.predict_proba(X_test_final)[:, 1]

# Crear el DataFrame para enviar a Kaggle
submission = pd.DataFrame({
    'Id': df_test_clean['Id'],
    'Probability': probabilidades_test
})

# Guardar el CSV
submission.to_csv('my_xgboost_submission.csv', index=False)
print("Archivo 'my_xgboost_submission.csv' generado con éxito y listo para Kaggle.")

import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Definimos la "función objetivo" que Optuna intentará maximizar
def objective(trial):
    # 1. Sugerimos un rango de hiperparámetros para que Optuna explore
    #    NOTA: scale_pos_weight se calcula dinámicamente por pliegue (ver abajo)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1 # Usa todos los núcleos de tu procesador
    }
    
    # 2. Validación Cruzada (Cross-Validation) sobre datos PRE-imputación
    # La imputación se ajusta POR PLIEGUE para evitar fuga de datos.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(X_cv, y_cv):
        X_fold_train = X_cv.iloc[train_idx].copy()
        X_fold_val = X_cv.iloc[val_idx].copy()
        y_fold_train, y_fold_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
        
        # Imputación POR PLIEGUE: fit SOLO en fold_train, transform en ambos
        fold_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            max_iter=10, random_state=42, sample_posterior=False, skip_complete=True
        )
        X_fold_train[cols_impute] = fold_imputer.fit_transform(
            X_fold_train[cols_impute].astype('float64')
        )
        X_fold_val[cols_impute] = fold_imputer.transform(
            X_fold_val[cols_impute].astype('float64')
        )
        
        # Peso de balanceo DINÁMICO por pliegue (refleja la proporción real del fold)
        fold_peso = y_fold_train.value_counts()[0] / y_fold_train.value_counts()[1]
        params['scale_pos_weight'] = fold_peso
        
        # Entrenamos el modelo con los parámetros sugeridos
        model = xgb.XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train, verbose=False)
        
        # Evaluamos
        pred_proba = model.predict_proba(X_fold_val)[:, 1]
        auc_scores.append(roc_auc_score(y_fold_val, pred_proba))
        
    # Devolvemos el promedio del AUC de los 5 cortes
    return np.mean(auc_scores)

# 3. Ejecutamos el estudio de Optuna
print("Iniciando la cacería de los mejores hiperparámetros...")
# direction='maximize' porque queremos el AUC más alto posible
study = optuna.create_study(direction='maximize') 

# n_trials=20 es un buen inicio. En producción real podrías dejarlo en 100.
study.optimize(objective, n_trials=20) 

print("\n=== LOS MEJORES HIPERPARÁMETROS ENCONTRADOS ===")
print(study.best_params)
print(f"Mejor ROC-AUC logrado: {study.best_value:.4f}")

# 4. Entrenar el modelo final con estos súper parámetros
# Peso de balanceo calculado sobre la TOTALIDAD de los datos de entrenamiento
peso_balanceo_final = y.value_counts()[0] / y.value_counts()[1]

best_xgb_model = xgb.XGBClassifier(
    **study.best_params, 
    scale_pos_weight=peso_balanceo_final,
    random_state=42
)
best_xgb_model.fit(X, y)
print("\nModelo final reentrenado y listo para producción.")


# ── GUARDAR EL MODELO FINAL (artefacto Docker) ──
def save_final_model(model, output_dir='models', filename='xgboost_final.json'):
    """
    Persiste el modelo XGBoost entrenado con los parámetros óptimos.
    Usa el método nativo save_model() de XGBoost (formato JSON) para evitar
    dependencias de entorno: no se rompe si la versión de Python o scikit-learn
    difiere entre tu máquina local y el contenedor Docker final.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    model.save_model(output_path)
    print(f"✅ Modelo final guardado en: '{output_path}'")
    return output_path

save_final_model(best_xgb_model)

