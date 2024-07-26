"""
En la investigación de tesis de maestría realizada por el estudiante Mohammed Ismail Lifta (2023-2024) en el Departamento de Ciencias de la 
Computación, Facultad de Ciencias de la Computación y Matemáticas de la Universidad de Tikrit, Irak, se recopilaron datos del Laboratorio de 
Agricultura sobre plantas que crecen en un invernadero de IoT. e invernadero tradicional. El estudio fue supervisado por el profesor (asistente) 
Wisam Dawood Abdullah, administrador de Cisco Networking Academy/Universidad de Tikrit.

Descripción del conjunto de datos
El conjunto de datos "Advanced_IoT_Dataset.csv" consta de 30.000 entradas y 14 columnas. A continuación se muestran las descripciones detalladas de 
cada columna:

Aleatorio: un identificador para cada registro, que probablemente indique una muestra o lote aleatorio (tipo de objeto).
Promedio de clorofila en la planta (ACHP): El contenido promedio de clorofila en la planta (tipo flotador).
Tasa de altura de la planta (PHR): La tasa de crecimiento en altura de la planta (tipo flotador).
Peso húmedo promedio del crecimiento vegetativo (AWWGV): El peso húmedo promedio del crecimiento vegetativo (tipo flotador).
Área foliar promedio de la planta (ALAP): El área foliar promedio de la planta (tipo flotador).
Número promedio de hojas de planta (ANPL): El número promedio de hojas por planta (tipo flotador).
Diámetro promedio de raíz (ARD): El diámetro promedio de las raíces de la planta (tipo flotador).
Peso seco promedio de la raíz (ADWR): El peso seco promedio de las raíces de la planta (tipo flotador).
Porcentaje de materia seca para crecimiento vegetativo (PDMVG): El porcentaje de materia seca en crecimiento vegetativo (tipo flotador).
Longitud promedio de raíz (ARL): La longitud promedio de las raíces de la planta (tipo flotador).
Peso húmedo promedio de la raíz (AWWR): El peso húmedo promedio de las raíces de la planta (tipo flotador).
Peso seco promedio de plantas vegetativas (ADWV): El peso seco promedio de las partes vegetativas de la planta (tipo flotador).
Porcentaje de materia seca para el crecimiento de las raíces (PDMRG): El porcentaje de materia seca en el crecimiento de las raíces (tipo 
flotador).
Clase: La clase o categoría a la que pertenece el registro de planta (tipo de objeto).

Descripción más detallada de las columnas del conjunto de datos:
Aleatorio: un identificador categórico para cada registro. Esta columna parece tener valores como R1, R2 y R3, que podrían representar 
diferentes 
muestras aleatorias.

Promedio de clorofila en la planta (ACHP): esta columna contiene valores flotantes que representan el contenido promedio de clorofila en la 
planta. 
La clorofila es vital para la fotosíntesis y su medición puede indicar la salud y la eficiencia de la planta para convertir la energía 
luminosa en energía química.

Tasa de altura de la planta (PHR): esta columna contiene valores flotantes que representan la tasa de crecimiento en altura de la planta. Esta 
métrica es esencial para comprender la dinámica de crecimiento vertical de la planta a lo largo del tiempo.

Peso húmedo promedio del crecimiento vegetativo (AWWGV): esta columna contiene valores flotantes que representan el peso húmedo promedio de las 
partes vegetativas de la planta. El peso húmedo puede ser un indicador del contenido de agua y de la biomasa total del crecimiento vegetativo 
de la planta.

Área foliar promedio de la planta (ALAP): Esta columna contiene valores flotantes que representan el área foliar promedio de la planta. El área 
foliar es un factor crítico en la fotosíntesis, ya que determina la superficie disponible para la absorción de luz.

Número promedio de hojas de plantas (ANPL): esta columna contiene valores flotantes que representan el número promedio de hojas por planta. La 
cantidad de hojas puede correlacionarse con la capacidad de la planta para realizar la fotosíntesis y su salud general.

Diámetro promedio de la raíz (ARD): esta columna contiene valores flotantes que representan el diámetro promedio de las raíces de la planta. El 
diámetro de la raíz puede afectar la capacidad de la planta para absorber agua y nutrientes del suelo.

Peso seco promedio de la raíz (ADWR): Esta columna contiene valores flotantes que representan el peso seco promedio de las raíces de la planta. 
El peso seco es una medida de la biomasa de la planta después de eliminar el contenido de agua y es un indicador de la capacidad estructural 
y de almacenamiento de la raíz.

Porcentaje de materia seca para el crecimiento vegetativo (PDMVG): Esta columna contiene valores flotantes que representan el porcentaje de 
materia seca en las partes vegetativas de la planta. Esta métrica indica la proporción de la biomasa de la planta que no es agua, lo que 
puede ser crucial para comprender su estado estructural y nutricional.

Longitud promedio de la raíz (ARL): esta columna contiene valores flotantes que representan la longitud promedio de las raíces de la planta. 
La longitud de las raíces puede influir en la capacidad de la planta para explorar y absorber nutrientes y agua del suelo.

Peso húmedo promedio de la raíz (AWWR): esta columna contiene valores flotantes que representan el peso húmedo promedio de las raíces de la 
planta. 
El peso húmedo incluye el contenido de agua en las raíces, lo que indica su biomasa general y su capacidad de retención de agua.

Peso seco promedio de plantas vegetativas (ADWV): esta columna contiene valores flotantes que representan el peso seco promedio de las partes 
vegetativas de la planta. Esta medida refleja la biomasa estructural de la planta sin contenido de agua.

Porcentaje de materia seca para el crecimiento de las raíces (PDMRG): Esta columna contiene valores flotantes que representan el porcentaje de 
materia seca en las raíces de la planta. Esta métrica muestra la proporción de biomasa de raíces que no es agua, importante para evaluar la 
salud y función de las raíces.

Clase: Columna categórica que indica la clase o categoría a la que pertenece el registro de la planta. Puede representar diferentes grupos o 
condiciones en las que se estudiaron o clasificaron las plantas.

El conjunto de datos proporciona información completa sobre diversas métricas de plantas relacionadas con el crecimiento vegetativo y radicular, 
junto con etiquetas de clasificación que podrían usarse para fines de análisis o aprendizaje automático."""

#Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import plotly.express as px
from datetime import datetime
from tabulate import tabulate



#Cargar el conjunto de datos
df = pd.read_csv('Advanced_IoT_Dataset.csv')

#Información básica
print(df)

print(df.info())

df.isnull().sum()

print(df.describe())

print(df)

print(df.head())

print(df.describe().T)

df.describe().T.plot(kind ='bar')

print(df.columns)

df.isna().sum()

sns.heatmap(df.isna())

df.duplicated().sum()

df['Class'].unique() 

df['Class'].value_counts()

#Distribución de cada característica
num_columns = len(df.columns)

ncols = 3
nrows = (num_columns + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

for i, column in enumerate(df.columns):
    row = i // ncols
    col = i % ncols
    sns.histplot(df[column], kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribución de {column}')

for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()

#Gráfico de pares para relaciones entre características
sns.pairplot(df, hue='Class')
plt.show()

#Codificar categórico
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])
df['Random'] = label_encoder.fit_transform(df['Random'])

#Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='cool')
plt.title('Matriz de correlación\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Funciones separadas y variable de destino
X = df.drop('Class', axis=1)
y = df['Class']

#dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Selección y entrenamiento de modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'--- {model_name} ---')
    print('Exactitud:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
#Ajuste de modelo con GridSearchCV
param_grid = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}
best_model_name = 'RandomForest'
best_model = RandomForestClassifier()
grid_search = GridSearchCV(best_model, param_grid[best_model_name], cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('Mejores parámetros:', grid_search.best_params_)
best_model = grid_search.best_estimator_

#Evaluación final
y_pred = best_model.predict(X_test)
print('Evaluación final del modelo:')
print('Exactitud:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#guardar el modelo

joblib.dump(best_model, 'best_model.pkl')

numeric_cols = df.select_dtypes(include=np.number).columns  #Elija solo columnas numéricas
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación\n', fontsize = '16', fontweight = 'bold')
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

#Predecir valores usando el modelo
y_pred = model.predict(X_test)

#Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo en datos de prueba: {accuracy}')

## Aquí importando algunas bibliotecas.
# para gestionar datos

# para organizar la impresión

#Predicción de la clasificación de plantas
#¡Los modelos complejos no siempre son necesarios! ver las conclusiones finales

#Observación:
#Me parece que las clases están ordenadas, para un modelo más generalista una buena solución podría ser barajar los datos

shuffled_df = df.sample(frac=1).reset_index(drop=True)

"""Clase MultiModelEvaluator
La clase MultiModelEvaluator está diseñada para facilitar la evaluación de múltiples modelos de aprendizaje automático para una tarea de clasificación. Proporciona
métodos para realizar entrenamiento de modelos, ajuste de hiperparámetros mediante búsqueda de cuadrícula y evaluación de los mejores modelos utilizando varias métricas de clasificación.

Descripción general
La clase MultiModelEvaluator encapsula las siguientes funcionalidades:

Cargando y preprocesando el conjunto de datos.
Codificación de la variable de destino usando OneHotEncoder.
Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
Definir un conjunto de modelos de clasificación a evaluar.
Construcción de canalizaciones para cada modelo para agilizar el preprocesamiento y el ajuste del modelo.
Especificación de cuadrículas de parámetros para el ajuste de hiperparámetros.
Realizar una validación cruzada de búsqueda de cuadrícula para encontrar los mejores modelos.
Evaluar los mejores modelos utilizando métricas como exactitud, precisión, recuperación y puntuación F1. Esta clase proporciona un enfoque 
sistemático para comparar el rendimiento de diferentes algoritmos de clasificación en un conjunto de datos determinado. Permite experimentar 
fácilmente con varios modelos y hiperparámetros, que ayudan a identificar el enfoque más eficaz para la tarea de clasificación en cuestión."""
class MultiModelEvaluator:
    
    def __init__(self, df):
        # Establecer la característica que desea predecir
        self.X = df.drop(columns=['Class'])
        self.y = df['Class']
        
        # Codifique la etiqueta usando OneHotEncoder
        self.label_encoder = OneHotEncoder(sparse_output=False)
        self.y_label       = self.label_encoder.fit_transform(self.y.values.reshape(-1, 1))
        
        # Divida los datos en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y_label,
            test_size=0.2,
            random_state=42
        )

        # Definir modelos
        self.models = {
            'KNN': KNeighborsClassifier(),
            'XGBoost': XGBClassifier()
        }

        # Crear pipelines para cada modelo
        self.pipelines = self.create_pipelines()

        # Definir cuadrículas de parámetros para cada modelo
        self.param_grids = {
            'KNN': {
                'model__n_neighbors': [3, 5],
                'model__weights': ['distance']
            },
            'XGBoost': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2]
            }
        }

        # Definir métricas de evaluación
        self.scoring = {
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }

        # Realice GridSearchCV para cada modelo
        self.best_models = {}
        
    def create_pipelines(self):
        return {
            name: Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            for name, model in self.models.items()
        }

    def find_best_models(self):
        for name, pipeline in self.pipelines.items():
            grid_search = GridSearchCV(
                pipeline,
                self.param_grids[name],
                scoring='accuracy',
                refit='mse',
                cv=KFold(n_splits=5,shuffle=True,random_state=42),
                error_score='raise'
            )
            grid_search.fit(self.X_train, self.y_train)
            self.best_models[name] = grid_search.best_estimator_

    def evaluate_best_models(self):
        results = []
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

        for name, model in self.best_models.items():
            predicted_values = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, predicted_values)
            precision = precision_score(self.y_test, predicted_values, average='weighted', zero_division=1)
            recall = recall_score(self.y_test, predicted_values, average='weighted')
            f1 = f1_score(self.y_test, predicted_values, average='weighted')

            results.append([name, accuracy, precision, recall, f1])

        print(tabulate(results, headers=headers, tablefmt="pretty"))
mult_model_eval = MultiModelEvaluator(shuffled_df)
mult_model_eval.find_best_models()
mult_model_eval.evaluate_best_models()

"""Conclusión
Exploramos el rendimiento de los modelos KNN (K-Nearest Neighbors) y XGBoost en nuestro conjunto de datos. Sorprendentemente, ambos modelos
logró una notable puntuación de precisión de 1,0, lo que indica su rendimiento excepcional en la clasificación de los datos.

Resultados clave:
Precisión perfecta: Tanto el modelo KNN como el XGBoost lograron una puntuación de precisión perfecta de 1,0, lo que sugiere que pudieron 
clasificar perfectamente el puntos de datos en nuestro conjunto de datos.

Características de los datos: las altas puntuaciones de precisión implican que el conjunto de datos podría poseer patrones o grupos claros que 
estos modelos pudieron capturar de manera efectiva.

Complejidad del modelo: a pesar de la simplicidad del algoritmo KNN, funcionó igualmente bien en comparación con el modelo XGBoost más complejo. 
Esto indica que para nuestro conjunto de datos, los modelos complejos pueden no ser necesarios.

Recomendaciones:
Selección de modelo: según nuestros hallazgos, se puede considerar KNN o XGBoost para tareas de clasificación en conjuntos de datos similares. 
Sin embargo, considerando Dada la simplicidad y la interpretabilidad de KNN, podría preferirse si los recursos computacionales son limitados 
o la interpretabilidad es crucial.
En conclusión, nuestra exploración demuestra la efectividad de los modelos KNN y XGBoost en nuestro conjunto de datos, destacando la 
importancia de comprender las características de los datos y seleccionar enfoques de modelado apropiados."""