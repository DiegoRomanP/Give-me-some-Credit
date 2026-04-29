# 📊 Give Me Some Credit — Modelo de Scoring Crediticio

> **Predicción de incumplimiento crediticio** utilizando XGBoost, optimización bayesiana con Optuna y explicabilidad con SHAP.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Descripción del Proyecto

Este proyecto aborda uno de los problemas más críticos de la industria financiera: **¿cómo predecir si un prestatario incurrirá en un incumplimiento grave (morosidad de 90+ días) en los próximos dos años?**

Utilizando datos de la competencia [Give Me Some Credit](https://www.kaggle.com/competitions/give-me-some-credit) de Kaggle, se construyó un modelo de *credit scoring* de extremo a extremo que incluye:

- **Análisis exploratorio** profundo con detección de outliers y valores nulos.
- **Limpieza de datos** con imputación multivariada (MICE) y transformaciones logarítmicas.
- **Modelado predictivo** con XGBoost y dos estrategias de balanceo de clases.
- **Optimización de hiperparámetros** con Optuna (búsqueda bayesiana).
- **Explicabilidad** del modelo con SHAP para transparencia en decisiones crediticias.

### 🎯 Impacto del Negocio

En el sector financiero, un modelo de *scoring* preciso tiene un impacto directo en:

| Beneficio | Descripción |
|-----------|-------------|
| **Reducción de pérdidas** | Identificar prestatarios de alto riesgo antes de aprobar créditos |
| **Automatización** | Reemplazar procesos manuales de evaluación crediticia |
| **Cumplimiento regulatorio** | SHAP proporciona explicaciones individuales para cada decisión |
| **Optimización de cartera** | Mejor asignación de tasas de interés según nivel de riesgo |

---

## 🗂️ Estructura del Proyecto

```
Give-Me-Some-Credit/
├── data/                              # Datasets (train.csv, test.csv)
├── images_result/                     # Visualizaciones generadas
│   ├── datos_box_plot.png             # Boxplots antes de limpieza
│   ├── datos_box_plot_after_cleaning.png  # Boxplots después de limpieza
│   ├── datos_histograma.png           # Histogramas originales
│   ├── datos_histograma_norm.png      # Histogramas normalizados
│   ├── datos_histograma_plot_after_cleaning.png  # Histogramas post-limpieza
│   └── Comparativa_curva_roc.png      # Curva ROC comparativa
├── models/                            # Modelos entrenados
│   ├── xgboost_base_optimizado.json   # XGBoost con pesos algorítmicos
│   └── xgboost_smote_optimizado.json  # XGBoost con SMOTE
├── Give_me_some_credit.ipynb          # Notebook principal del análisis
├── imputador_mice_financiero.joblib   # Imputador MICE entrenado
├── pyproject.toml                     # Configuración del proyecto (uv)
├── uv.lock                           # Lock de dependencias
├── .python-version                    # Versión de Python (3.14)
└── README.md                          # Este archivo
```

---

## ⚙️ Configuración del Entorno

Este proyecto utiliza [**uv**](https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales.

### Requisitos Previos

- **Python 3.14+** instalado en tu sistema.
- **uv** instalado globalmente:
  ```bash
  pip install uv
  ```

### Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/Give-Me-Some-Credit.git
   cd Give-Me-Some-Credit
   ```

2. **Crear el entorno virtual e instalar dependencias:**
   ```bash
   uv sync
   ```
   > Esto leerá el `pyproject.toml` y el `uv.lock` para instalar exactamente las mismas versiones de todas las dependencias.

3. **Descargar los datos:**

   Los datos deben colocarse en la carpeta `data/`. Puedes descargarlos desde:
   - [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/give-me-some-credit/data)

   O usar la API de Kaggle:
   ```bash
   pip install kagglehub
   ```
   ```python
   import kagglehub
   path = kagglehub.competition_download('give-me-some-credit')
   ```

4. **Ejecutar el notebook:**
   ```bash
   uv run jupyter notebook Give_me_some_credit.ipynb
   ```

### Dependencias Principales

| Paquete | Versión | Uso |
|---------|---------|-----|
| `pandas` | ≥3.0.2 | Manipulación de DataFrames |
| `numpy` | ≥2.4.4 | Operaciones numéricas |
| `matplotlib` | ≥3.10.9 | Visualización de gráficos |
| `seaborn` | ≥0.13.2 | Visualización estadística |
| `scikit-learn` | ≥1.8.0 | Preprocesamiento, métricas e imputación MICE |
| `xgboost` | ≥3.2.0 | Modelo de clasificación |
| `optuna` | ≥4.8.0 | Optimización bayesiana de hiperparámetros |
| `shap` | ≥0.51.0 | Explicabilidad del modelo |
| `imbalanced-learn` | ≥0.14.1 | Sobremuestreo con SMOTE |

---

## 🔍 Análisis Exploratorio (EDA)

### Distribución de Variables — Boxplots

Los boxplots revelan la presencia de **outliers extremos** en múltiples variables del dataset original:

<p align="center">
  <img src="images_result/datos_box_plot.png" alt="Boxplots antes de la limpieza" width="90%">
</p>

**Hallazgos principales:**
- **`age`**: Registros con edad = 0 (error de datos).
- **Variables de retraso** (`NumberOfTime30-59DaysPastDueNotWorse`, etc.): Valores de 96 y 98, imposibles en una ventana de 24 meses.
- **`RevolvingUtilizationOfUnsecuredLines`** y **`DebtRatio`**: Distribuciones con colas extremadamente largas.

### Distribución de Variables — Histogramas

<p align="center">
  <img src="images_result/datos_histograma.png" alt="Histogramas originales" width="90%">
</p>

Los histogramas confirman el **sesgo extremo** en las variables financieras, lo que justifica la aplicación de transformaciones logarítmicas.

### Distribuciones Normalizadas

Tras aplicar `log1p` a las variables sesgadas:

<p align="center">
  <img src="images_result/datos_histograma_norm.png" alt="Histogramas normalizados" width="90%">
</p>

### Datos Después de la Limpieza

Boxplots y distribuciones tras el preprocesamiento completo (capping, imputación MICE, transformaciones):

<p align="center">
  <img src="images_result/datos_box_plot_after_cleaning.png" alt="Boxplots después de limpieza" width="90%">
</p>

<p align="center">
  <img src="images_result/datos_histograma_plot_after_cleaning.png" alt="Histogramas post-limpieza" width="90%">
</p>

> Los outliers extremos fueron controlados y las distribuciones se normalizaron significativamente.

---

## 📈 Resultados del Modelado

Se compararon dos estrategias para manejar el desbalanceo de clases:

### Estrategia 1: XGBoost con Pesos Algorítmicos

Utiliza `scale_pos_weight` para penalizar más los errores en la clase minoritaria (impagos).

**Reporte de clasificación (evaluación local — 20% holdout):**

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Buen Pagador (0) | 0.98 | 0.81 | 0.88 | 19,571 |
| Impago (1) | 0.22 | **0.77** | 0.34 | 1,390 |
| **Accuracy** | | | **0.80** | 20,961 |

- **ROC-AUC Local:** `0.8664`
- **ROC-AUC optimizado con Optuna (CV):** `0.8643`

**Hiperparámetros óptimos (Optuna):**
| Parámetro | Valor |
|-----------|-------|
| `n_estimators` | 239 |
| `max_depth` | 4 |
| `learning_rate` | 0.0449 |
| `subsample` | 0.6427 |
| `colsample_bytree` | 0.6226 |

### Estrategia 2: XGBoost con SMOTE

Genera muestras sintéticas de la clase minoritaria antes del entrenamiento.

**Reporte de clasificación (evaluación local — 20% holdout):**

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Buen Pagador (0) | 0.96 | 0.96 | 0.96 | 19,571 |
| Impago (1) | 0.42 | 0.37 | 0.39 | 1,390 |
| **Accuracy** | | | **0.92** | 20,961 |

- **ROC-AUC Local:** `0.8444`
- **ROC-AUC optimizado con Optuna (CV):** `0.8501`

**Hiperparámetros óptimos (Optuna):**
| Parámetro | Valor |
|-----------|-------|
| `n_estimators` | 216 |
| `max_depth` | 3 |
| `learning_rate` | 0.0379 |
| `subsample` | 0.7426 |
| `colsample_bytree` | 0.8220 |

### Comparativa Visual — Curva ROC

<p align="center">
  <img src="images_result/Comparativa_curva_roc.png" alt="Comparativa Curva ROC" width="70%">
</p>

La curva ROC demuestra que el modelo con **pesos algorítmicos** domina al modelo con SMOTE en **todos los umbrales de clasificación**.

### Tabla Comparativa Final

| Métrica | XGBoost (Pesos Algorítmicos) | XGBoost (SMOTE) |
|---------|:----------------------------:|:---------------:|
| **ROC-AUC (CV)** | **0.8643** ✅ | 0.8501 |
| **Recall (Impago)** | **0.77** ✅ | 0.37 |
| **Precision (Impago)** | 0.22 | **0.42** ✅ |
| **Accuracy** | 0.80 | **0.92** ✅ |

> **Nota:** En el contexto de *scoring* crediticio, el **Recall** de la clase de impago es la métrica más importante, ya que el costo de no detectar un impago (falso negativo) es significativamente mayor que el de rechazar un buen pagador (falso positivo). Por esta razón, el modelo con **pesos algorítmicos** es el ganador.

---

## 🏆 Conclusiones

### Modelo Recomendado

El **XGBoost con Pesos Algorítmicos**, optimizado con Optuna, es el modelo seleccionado para producción por las siguientes razones:

1. **Superior ROC-AUC (0.8643):** Mejor capacidad discriminativa global.
2. **Recall de impago del 77%:** Detecta 3 de cada 4 clientes que incumplirán, lo cual es crítico para la gestión de riesgo.
3. **Sin datos sintéticos:** Trabaja exclusivamente con datos reales, evitando el ruido que SMOTE puede introducir.
4. **Mayor eficiencia:** No requiere un paso adicional de generación de muestras.

### ¿Por qué Pesos Algorítmicos supera a SMOTE?

| Aspecto | Pesos Algorítmicos | SMOTE |
|---------|-------------------|-------|
| **Datos utilizados** | 100% datos reales | Datos reales + sintéticos |
| **Ruido introducido** | Ninguno | Muestras sintéticas pueden introducir ruido |
| **Recall (Impago)** | **0.77** | 0.37 |
| **Generalización** | Mejor | Puede sobreajustar a patrones sintéticos |

### Recomendaciones para Producción

1. **Monitoreo continuo:** Implementar detección de *data drift* en las distribuciones de entrada.
2. **Reentrenamiento periódico:** Actualizar el modelo con datos recientes al menos trimestralmente.
3. **Explicabilidad en tiempo real:** Utilizar SHAP para generar explicaciones individuales por cada decisión de crédito, cumpliendo con regulaciones de transparencia.
4. **Umbral personalizado:** Ajustar el threshold de clasificación según la política de riesgo de la institución financiera.

---

## 👤 Autor

**Diego Roman**

---

*Proyecto desarrollado con enfoque de Data Analytics, priorizando reproducibilidad, interpretabilidad y rigor estadístico.*
