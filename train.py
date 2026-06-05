import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

df_train_clean = pd.read_csv('data/train_clean.csv')
df_test_clean = pd.read_csv('data/test_clean.csv')

df_train_clean = df_train_clean.dropna(subset=['SeriousDlqin2yrs'])

# Separar Features (X) y Target (y) del TRAIN LIMPIO
X = df_train_clean.drop(columns=['SeriousDlqin2yrs', 'Id'])
y = df_train_clean['SeriousDlqin2yrs']

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
    use_label_encoder=False,
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
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': peso_balanceo, # Mantenemos el peso que calculamos antes
        'eval_metric': 'auc',
        'random_state': 42,
        'use_label_encoder': False,
        'n_jobs': -1 # Usa todos los núcleos de tu procesador
    }
    
    # 2. Validación Cruzada (Cross-Validation)
    # Entrenamos en 5 cortes diferentes para asegurar que el modelo no memorice
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
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
best_xgb_model = xgb.XGBClassifier(
    **study.best_params, 
    scale_pos_weight=peso_balanceo,
    random_state=42
)
best_xgb_model.fit(X, y)
print("\nModelo final reentrenado y listo para producción.")


# Guardar el modelo
best_xgb_model.save_model('xgboost_base_optimizado.json')
print("✅ Modelo XGBoost guardado: 'xgboost_base_optimizado.json'")

