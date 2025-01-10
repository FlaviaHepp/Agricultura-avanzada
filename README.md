# Agricultura-avanzada
Proyecto: Análisis de Agricultura Avanzada con IoT y Modelos Predictivos

**Origen**
El dataset se obtuvo de la tesis de maestría realizada por el estudiante Mohammed Ismail Lifta (2023-2024) del Departamento de Ciencias de la 
Computación, Facultad de Ciencias de la Computación y Matemáticas de la Universidad de Tikrit, Irak.

**Descripción de las columnas:**
*Aleatorio:* un identificador para cada registro, que probablemente indique una muestra o lote aleatorio (tipo de objeto).
*Promedio de clorofila en la planta (ACHP):* El contenido promedio de clorofila en la planta (tipo flotador).
*Tasa de altura de la planta (PHR):* La tasa de crecimiento en altura de la planta (tipo flotador).
*Peso húmedo promedio del crecimiento vegetativo (AWWGV):* El peso húmedo promedio del crecimiento vegetativo (tipo flotador).
*Área foliar promedio de la planta (ALAP):* El área foliar promedio de la planta (tipo flotador).
*Número promedio de hojas de planta (ANPL):* El número promedio de hojas por planta (tipo flotador).
*Diámetro promedio de raíz (ARD):* El diámetro promedio de las raíces de la planta (tipo flotador).
*Peso seco promedio de la raíz (ADWR):* El peso seco promedio de las raíces de la planta (tipo flotador).
*Porcentaje de materia seca para crecimiento vegetativo (PDMVG):* El porcentaje de materia seca en crecimiento vegetativo (tipo flotador).
*Longitud promedio de raíz (ARL):* La longitud promedio de las raíces de la planta (tipo flotador).
*Peso húmedo promedio de la raíz (AWWR):* El peso húmedo promedio de las raíces de la planta (tipo flotador).
*Peso seco promedio de plantas vegetativas (ADWV):* El peso seco promedio de las partes vegetativas de la planta (tipo flotador).
*Porcentaje de materia seca para el crecimiento de las raíces (PDMRG):* El porcentaje de materia seca en el crecimiento de las raíces (tipo 
flotador).
*Clase:* La clase o categoría a la que pertenece el registro de planta (tipo de objeto).

**Desarrollo**
Estudio del impacto de sistemas agrícolas avanzados basados en IoT en invernaderos comparados con métodos tradicionales. El proyecto incluyó el análisis de métricas vegetativas y radiculares de plantas para clasificar muestras y evaluar el rendimiento de diferentes técnicas agrícolas.
*Herramientas:* Python, pandas, seaborn, matplotlib, scikit-learn, XGBoost, KNN, GridSearchCV, Pipeline.
**Resultados clave:**
Clasificación de muestras con precisión perfecta (100%) mediante modelos KNN y XGBoost.
Identificación de patrones críticos en métricas como contenido de clorofila, área foliar y tasa de crecimiento.
Desarrollo de pipelines para preprocesamiento y ajuste de hiperparámetros, optimizando la eficiencia de los modelos.
*Habilidades aplicadas:* Análisis exploratorio de datos (EDA), preprocesamiento avanzado, evaluación de modelos con métricas (precisión, recall, F1), ajuste de hiperparámetros y visualización de datos.
