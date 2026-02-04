# 🌱Agricultura Avanzada con IoT y Machine Learning

Este proyecto implementa un sistema de análisis y clasificación basado en datos IoT para agricultura avanzada, utilizando técnicas de Análisis Exploratorio de Datos (EDA), Machine Learning supervisado y optimización de modelos para la toma de decisiones inteligentes en entornos agrícolas.

## 🎯Objetivo del proyecto

- Analizar datos provenientes de sensores IoT aplicados a agricultura.
- Explorar patrones, distribuciones y correlaciones entre variables ambientales.
- Clasificar estados o tipos de cultivos mediante modelos de Machine Learning.
- Comparar múltiples algoritmos de clasificación.
- Optimizar hiperparámetros y seleccionar el mejor modelo.
- Construir una base reutilizable para sistemas de agricultura de precisión.

## 📁Descripción del dataset

El conjunto de datos (Advanced_IoT_Dataset.csv) contiene mediciones simuladas/reales de sensores IoT agrícolas, incluyendo variables ambientales y una clase objetivo:
- Variables numéricas (sensores)
- Variables categóricas
- Class: etiqueta objetivo (estado/clase del cultivo o condición agrícola)
- Se incluyen características adicionales como:
  - Codificación de variables categóricas
  - Variables aleatorias para robustez del modelo
  
## 📊Análisis Exploratorio de Datos (EDA)

- Limpieza y validación
- Detección de valores faltantes
- Identificación de duplicados
- Análisis de tipos de datos
- Estadísticas descriptivas completas
- Visualización
- Histogramas con KDE por variable
- Gráficos de pares (pairplot) por clase
- Mapas de calor de valores faltantes
- Matrices de correlación
- Visualización comparativa de estadísticas descriptivas

## 🔄Preprocesamiento de datos

- Codificación de variables categóricas con LabelEncoder y OneHotEncoder
- Separación de variables predictoras y variable objetivo
- División entrenamiento / prueba
- Escalado de variables con StandardScaler
- Transformación a formato compatible con pipelines

## 🤖Modelos de Machine Learning

- Se entrenan y comparan múltiples modelos de clasificación:
- Modelos base
- Regresión Logística
- Random Forest Classifier
- Gradient Boosting Classifier
- Modelos avanzados
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

## ⚙️Optimización y evaluación 

- GridSearchCV para ajuste de hiperparámetros
- Validación cruzada con K-Fold
- Métricas de evaluación:
  -- Accuracy
  -- Precision
  -- Recall
  -- F1 Score
  -- MAE / MSE (en evaluaciones extendidas)

## 🧩Arquitectura avanzada

- Clase MultiModelEvaluator
- Se implementa una clase personalizada que permite:
- Entrenar múltiples modelos automáticamente
- Crear pipelines de preprocesamiento y modelado
- Ajustar hiperparámetros
- Evaluar modelos con métricas estándar
- Comparar resultados de forma sistemática

Este enfoque facilita la reutilización y escalabilidad del proyecto.

## 💾Persistencia del modelo

- Exportación del mejor modelo entrenado mediante joblib
- Listo para integración en sistemas productivos o APIs

## 🛠️Tecnologías utilizadas

- Python
- pandas / numpy
- Matplotlib / Seaborn
- `Plotly`
- scikit-learn
- XGBoost
- `joblib`
- `Machine Learning aplicado a IoT`

## 📂Estructura del proyecto

├── Agricultura avanzada IoT.py
├── Advanced_IoT_Dataset.csv
├── best_model.pkl
└── README.md


## 📌Resultados principales

- Alta separabilidad entre clases del dataset.
- Modelos KNN y XGBoost alcanzan precisión perfecta en el conjunto de prueba.
- El dataset presenta patrones claros aprovechables por modelos supervisados.
- Modelos simples pueden ser tan efectivos como modelos complejos.
- Pipeline escalable para proyectos de Smart Farming.

## 🌍Aplicaciones reales

- Monitoreo inteligente de cultivos
- Agricultura de precisión
- Sistemas de alerta temprana
- Optimización de recursos (agua, fertilizantes)
- Integración con plataformas IoT y dashboards

## ⚠️Disclaimer

Este proyecto tiene fines educativos y demostrativos.
No sustituye sistemas agrícolas productivos sin validación en campo.

##👤 Autor

Flavia Hepp
Data Science en formación· Machine Learning · IoT · Agricultura Inteligente
