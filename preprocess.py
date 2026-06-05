import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Descargar la versión más reciente del dataset
path = kagglehub.competition_download('give-me-some-credit')

print("Path to competition files:", path)

# ubicacion de los datos
path_train = "data/train.csv"
path_test = "data/test.csv"

# cargamos los datos
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

# Aplicacion de Inputacion Multivariada(MICE por su siglas en ingles)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 1. Seleccionar las columnas involucradas en la imputación
cols_impute = ['MonthlyIncome', 'age', 'DebtRatio', 'NumberOfDependents']
df_subset = df_train_clean[cols_impute].copy()

# 2. Convertir a float64 y pasar <NA> (pandas) a np.nan (numpy/sklearn)
# IterativeImputer requiere matrices numéricas sin el tipo nullable de pandas
df_subset = df_subset.astype('float64')

# 3. Configurar MICE con un estimador robusto para datos tabulares
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    max_iter=10,          # Número de rondas de imputación (10 suele ser suficiente)
    random_state=42,      # Reproducibilidad
    sample_posterior=False, # False = imputación determinista (recomendado para ML)
    skip_complete=True    # Ignora filas sin nulos para acelerar
)

# 4. Ajustar y transformar
imputed_array = imputer.fit_transform(df_subset)

# 5. Reemplazar SOLO MonthlyIncome en el DataFrame original
df_train_clean['MonthlyIncome'] = imputed_array[:, 0]

# Verificación rápida
print("Nulos en MonthlyIncome después de imputar:", df_train_clean['MonthlyIncome'].isna().sum())
print("Rango de valores imputados:", df_train_clean['MonthlyIncome'].describe())


# 1. Seleccionar las columnas involucradas en la imputación
cols_impute = ['NumberOfDependents', 'age', 'DebtRatio', 'NumberOfDependents']
df_subset = df_train_clean[cols_impute].copy()

# 2. Convertir a float64 y pasar <NA> (pandas) a np.nan (numpy/sklearn)
# IterativeImputer requiere matrices numéricas sin el tipo nullable de pandas
df_subset = df_subset.astype('float64')

# 3. Ajustar y transformar
imputed_array = imputer.fit_transform(df_subset)

# 4. Reemplazar SOLO NumberOfDependents en el DataFrame original
df_train_clean['NumberOfDependents'] = imputed_array[:, 0]

# Verificación rápida
print("Nulos en NumberOfDependents después de imputar:", df_train_clean['NumberOfDependents'].isna().sum())
print("Rango de valores imputados:", df_train_clean['NumberOfDependents'].describe())