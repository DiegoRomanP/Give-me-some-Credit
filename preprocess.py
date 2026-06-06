import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Descargar la versión más reciente del dataset
path = kagglehub.competition_download('give-me-some-credit')
print("Path to competition files:", path)

# Asegurar que existe la carpeta para guardar los resultados limpios
os.makedirs('data', exist_ok=True)

# Ubicación de los datos descargados por kagglehub
path_train = os.path.join(path, "train.csv")
path_test = os.path.join(path, "test.csv")

# Cargamos los datos
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# Exploracion de datos
# eliminamos la ultima fila (encabezado duplicado detectado en el EDA)
df_train.drop(df_train.index[-1])

def convertir_tipos_datos(df):
    # Hacemos una copia para no modificar el DataFrame original
    df_conv = df.copy()
    
    # 1. Target: Y/N -> 1/0 (tipo entero pequeño, ideal para modelos de clasificación)
    # Se usa .astype(str) por si hay valores numéricos o mixtos en la columna
    
    df_conv['SeriousDlqin2yrs'] = (
        df_conv['SeriousDlqin2yrs'].astype(str).map({'1': 1, '0': 0})
        .astype('Int8')  # Entero de 8 bits con soporte para NaN
    )
    
    
    # 2. ID: convertir a numérico (se recomienda eliminarlo antes de entrenar modelos)
    df_conv['Id'] = pd.to_numeric(df_conv['Id'], errors='coerce').astype('Int64')
    
    # 3. Definir columnas según el diccionario
    float_cols = [
        'RevolvingUtilizationOfUnsecuredLines',
        'DebtRatio',
        'MonthlyIncome'
    ]
    
    int_cols = [
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]
    
    # 4. Conversión segura: strings/espacios/caracteres extra -> numérico, errores -> NaN
    all_numeric = float_cols + int_cols
    for col in all_numeric:
        df_conv[col] = pd.to_numeric(df_conv[col], errors='coerce')
        
    # 5. Asignar tipos finales optimizados
    for col in float_cols:
        df_conv[col] = df_conv[col].astype('float32')  # float32 ahorra ~50% de memoria vs float64
        
    for col in int_cols:
        df_conv[col] = df_conv[col].astype('Int32')    # Int32 es un "nullable integer" de pandas
        
    return df_conv

# Aplicar la función
# df = tu_dataframe_original
print("\nValores nulos por columna antes de convertir:")
print(df_train.isnull().sum())
try:
    df_train = convertir_tipos_datos(df_train)
    # Verificación rápida
    print("\nValores nulos por columna despues de convertir:")
    print(df_train.isnull().sum())
except Exception as e:
    print("No se pudo convertir los datos, se obtiene el error: \n", e)
    

# Tratamiento de los datos
df_train_clean = df_train.copy()
# Calculamos la mediana excluyendo los valores de 0
mediana_edad = df_train_clean.loc[df_train_clean['age'] > 0, 'age'].median()

# Reemplazamos los ceros con la mediana calculada
df_train_clean.loc[df_train_clean['age'] == 0, 'age'] = mediana_edad

print(f"Edad 0 reemplazada por la mediana: {mediana_edad}")


#capping con variables de retraso
# Definimos las columnas de retrasos
columnas_retrasos = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

# Reemplazamos valores mayores a 90 por NaN
for col in columnas_retrasos:
    # Contamos cuántos había para documentar
    cantidad_errores = len(df_train_clean[df_train_clean[col] > 90])
    df_train_clean.loc[df_train_clean[col] > 90, col] = np.nan
    print(f"Columna {col}: {cantidad_errores} códigos de error convertidos a NaN")

# Transformación logaritmica de variables con distribución sesgada
# Definimos las columnas con distribuciones de cola larga
columnas_log = [
    'MonthlyIncome', 
    'DebtRatio', 
    'RevolvingUtilizationOfUnsecuredLines'
]

# Aplicamos np.log1p creando nuevas columnas
for col in columnas_log:
    # Aplicamos la transformación. Nota: para MonthlyIncome que tiene NaN,
    # np.log1p mantendrá los NaN intactos para imputarlos después.
    df_train_clean[col] = np.log1p(df_train_clean[col])

#creacion del flag MonthlyIncome_Missing
df_train_clean['MonthlyIncome_Missing'] = df_train_clean['MonthlyIncome'].apply(lambda x: 1 if pd.isna(x) else 0)

# ── GUARDAR DATOS PRE-IMPUTACIÓN (con NaNs) ──
# train.py usará este archivo para validación cruzada, ajustando el imputer POR
# PLIEGUE (fold) y evitando así la fuga de datos entre pliegues de entrenamiento
# y validación. La imputación sobre el 100% del train solo se usa para el modelo
# final y para exportar el artefacto de producción.
df_train_clean.to_csv('data/train_pre_impute.csv', index=False)
print("📁 Datos pre-imputación guardados: 'data/train_pre_impute.csv'")

# Aplicacion de Inputacion Multivariada (MICE por sus siglas en inglés)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Columnas involucradas en la imputación (orden fijo: [0]=MonthlyIncome, [3]=NumberOfDependents)
cols_impute = ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents']

# Convertir a float64: IterativeImputer requiere matrices numéricas sin el tipo nullable de pandas
df_subset_train = df_train_clean[cols_impute].astype('float64')

# Configurar MICE con un estimador robusto para datos tabulares
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    max_iter=10,            # Número de rondas de imputación (10 suele ser suficiente)
    random_state=42,        # Reproducibilidad
    sample_posterior=False, # False = imputación determinista (recomendado para ML)
    skip_complete=True      # Ignora filas sin nulos para acelerar
)

# ── TRAIN: .fit() aprende las distribuciones SOLO del conjunto de entrenamiento ──
# Nunca usar fit_transform sobre el test; hacerlo filtra estadística del test al modelo.
imputer.fit(df_subset_train)
imputed_train = imputer.transform(df_subset_train)

# Devolver MonthlyIncome (col 0) y NumberOfDependents (col 3) al DataFrame de entrenamiento
df_train_clean['MonthlyIncome']      = imputed_train[:, 0]
df_train_clean['NumberOfDependents'] = imputed_train[:, 3]

# Verificación rápida
print("Nulos en MonthlyIncome después de imputar:",      df_train_clean['MonthlyIncome'].isna().sum())
print("Rango de MonthlyIncome imputado:",                df_train_clean['MonthlyIncome'].describe())
print("Nulos en NumberOfDependents después de imputar:", df_train_clean['NumberOfDependents'].isna().sum())
print("Rango de NumberOfDependents imputado:",           df_train_clean['NumberOfDependents'].describe())


def limpiar_dataset_financiero(df, fitted_imputer, mediana_edad_train=None):
    """
    Replica las transformaciones exactas del conjunto de entrenamiento sobre un nuevo dataset.

    Parámetros
    ----------
    df                : DataFrame de entrada (test / datos de producción desde Streamlit).
    fitted_imputer    : Objeto IterativeImputer ya ajustado sobre el train. Se usa SOLO
                        .transform() para evitar filtración de información estadística.
    mediana_edad_train: Mediana de edad calculada sobre el train. Si es None, se calcula
                        sobre df (solo válido en desarrollo, nunca en producción).
    """
    df_clean = df.copy()
    
    # 1. Tratar Edad (usamos la mediana del TRAIN para evitar Data Leakage)
    if mediana_edad_train is None:
        mediana_edad_train = df_clean.loc[df_clean['age'] > 0, 'age'].median()
    df_clean.loc[df_clean['age'] == 0, 'age'] = mediana_edad_train
    
    # 2. Tratar Códigos de Error
    columnas_retrasos = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    for col in columnas_retrasos:
        df_clean.loc[df_clean[col] > 90, col] = np.nan
        
    # 3. Transformaciones Logarítmicas
    columnas_log = ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']
    for col in columnas_log:
        df_clean[col] = np.log1p(df_clean[col])
        
    df_clean['MonthlyIncome_Missing'] = df_clean['MonthlyIncome'].apply(lambda x: 1 if pd.isna(x) else 0)

    # 4. Imputación: solo .transform() con el imputer ajustado en el train
    #    El orden de columnas debe coincidir exactamente con el usado en .fit()
    cols_impute = ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents']
    df_subset = df_clean[cols_impute].astype('float64')
    imputed_array = fitted_imputer.transform(df_subset)

    df_clean['MonthlyIncome']      = imputed_array[:, 0]
    df_clean['NumberOfDependents'] = imputed_array[:, 3]
    
    return df_clean

import joblib
import os

# Mediana de edad calculada SOLO sobre el train (sin tocar el test)
mediana_edad_referencia = df_train_clean.loc[df_train_clean['age'] > 0, 'age'].median()

# ── TEST: usamos el imputer ya entrenado (.transform únicamente) ──
df_test_clean = limpiar_dataset_financiero(
    df_test,
    fitted_imputer=imputer,
    mediana_edad_train=mediana_edad_referencia
)

# ── EXPORTAR ARTEFACTO DE PREPROCESAMIENTO ──
# El contenedor backend (y el frontend Streamlit) cargarán este .pkl para
# preparar los datos de usuarios finales con las mismas distribuciones del train.
os.makedirs('models', exist_ok=True)

preprocessor_artifact = {
    'imputer': imputer,                         # IterativeImputer ajustado sobre el train
    'mediana_edad_referencia': mediana_edad_referencia,  # Mediana de edad del train
    'cols_impute': ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents'],  # Orden fijo de columnas
    # ── GESTIÓN DEL ESPACIO LOGARÍTMICO ──
    # El imputer fue ajustado sobre datos ya transformados con np.log1p().
    # Cuando el backend/Streamlit reciba datos crudos del usuario (ej. salario
    # real), DEBE aplicar np.log1p() a estas columnas ANTES de llamar a
    # imputer.transform(). La función limpiar_dataset_financiero() ya incluye
    # este paso automáticamente.
    'columnas_log': ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines'],
    'nota_espacio_log': (
        'IMPORTANTE: El imputer espera datos en espacio log1p. Aplicar '
        'np.log1p() a las columnas MonthlyIncome, DebtRatio y '
        'RevolvingUtilizationOfUnsecuredLines ANTES de llamar a '
        'imputer.transform(). La función limpiar_dataset_financiero() '
        'ya incluye este paso.'
    ),
}

joblib.dump(preprocessor_artifact, 'models/preprocessor.pkl')
print("✅ Artefacto de preprocesamiento exportado: 'models/preprocessor.pkl'")

# ── GUARDAR CSVs LIMPIOS ──
df_train_clean.to_csv('data/train_clean.csv', index=False)
df_test_clean.to_csv('data/test_clean.csv', index=False)
print("✅ Datos limpios guardados en 'data/train_clean.csv' y 'data/test_clean.csv'")