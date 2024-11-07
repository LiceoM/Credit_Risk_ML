# Credit Card Default Risk - Riesgo de Incumplimiento de Tarjetas de Credito

El riesgo de incumplimiento de tarjetas de crédito es un desafío recurrente en la industria financiera. Este fenómeno se refiere a la probabilidad de que un titular de tarjeta de crédito sea incapaz o se niegue a pagar el saldo total o mínimo de su cuenta dentro del plazo establecido.

Este proyecto tiene como objetivo desarrollar distintos modelos de machine learning de clasificacion capaz de predecir con precisión la probabilidad de que un cliente incurra en un incumplimiento de pago. Para lograr esto, se llevará a cabo un análisis exhaustivo de un conjunto de datos históricos de clientes de tarjetas de crédito proporcionado por American Express para el concurso AmExpert CodeLab 2021, que puede encontrarse en kaggle siguiendo este enlace.

Predecir el riesgo de incumplimiento de tarjetas de crédito es crucial para las empresas de crédito por varias razones:

* Gestión de Riesgos: Permite a las instituciones financieras identificar a los clientes con mayor probabilidad de incumplimiento, lo que facilita la toma de decisiones informadas sobre la concesión de crédito y la gestión de carteras.
* Prevención de Pérdidas: Al identificar a los clientes de alto riesgo, las empresas pueden implementar medidas preventivas, como ajustar los límites de crédito o contactar a los clientes para ofrecer soluciones.
* Mejora de la Rentabilidad: Al reducir las pérdidas por incumplimiento, las empresas pueden mejorar su rentabilidad y fortalecer su posición en el mercado.
* Desarrollo de Productos Personalizados: Los modelos de predicción pueden utilizarse para desarrollar productos y servicios financieros personalizados que se adapten a las necesidades y riesgos específicos de cada cliente.

Este proyecto es un ejercicio academico en el contexto del curso de Machine Learning de la Universidad EAN. La intencion final es aplicar metodologias de aprendizaje automatico a un problema de clasificacion utilizando los modelos de clasificacion convencionales con tecnicas de seleccion de caracteristicas y optimizacion de hiperparametros.

# Metodologia

El proyecto se divide en los siguientes pasos:

1. Análisis exploratorio de los datos
2. Preprocesamiento de los datos
3. Optimización de hiperparámetros.
4. Entrenamiento de los modelos
5. Evaluación de los modelos
5. Comparacion de los modelos de clasificacion
9. Conclusiones

# Analisis Exploratorio de Datos (EDA) Dataset AmExpert CodeLab 2021

La metodologia de analisis exploratorio de datos consiste en los siguientes pasos:




## Preprocesamiento de datos

Se realizo un preprocesamiento de los datos para poder trabajar con ellos de una manera mas facil, se realizo un tratamiento de valores nulos, se eliminaron columnas que no aportaban informacion relevante y se transformaron las variables categoricas a numericas.

## Modelos de Machine Learning

Se probaron diferentes modelos de machine learning para determinar cual es el mejor modelo para este problema, los modelos que se probaron son los siguientes:

- Random Forest
- Catboost
- XgBoost
- LightGBM
- KNN
- Regresion Logistica

## Metricas de Evaluacion

Se utilizaron diferentes metricas de evaluacion para determinar cual es el mejor modelo, las metricas que se utilizaron son las siguientes:

## Matriz de Confusion

La matriz de confusión es una tabla que permite visualizar y evaluar el desempeño de un modelo de clasificación. Cada celda de la matriz representa una combinación de la clase real y la clase predicha por el modelo.

|           | Predicción Negativa | Predicción Positiva |
|-----------|---------------------|---------------------|
| Clase Negativa | Verdaderos Negativos (VN) | Falsos Positivos (FP) |
| Clase Positiva | Falso Negativos (FN) | Verdaderos Positivos (VP) |

- TP: Verdaderos Positivos (se predijo positivo y es positivo)
- TN: Verdaderos Negativos (se predijo negativo y es negativo)
- FP: Falsos Positivos (se predijo positivo pero es negativo)
- FN: Falsos Negativos (se predijo negativo pero es positivo)

###  Accuracy:

Representa la proporción de predicciones correctas sobre el total de predicciones. Es una medida general de la precisión del modelo, pero puede ser engañosa en conjuntos de datos desbalanceados.

Fórmula:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Precision:

Mide la proporción de predicciones positivas que son realmente correctas. Es útil cuando el costo de los falsos positivos es alto.

Fórmula:
Precision = TP / (TP + FP)

### Recall (Sensibilidad):

Mide la proporción de casos positivos que fueron correctamente identificados. Es útil cuando el costo de los falsos negativos es alto.

Fórmula:
Recall = TP / (TP + FN)

### F1-Score:

Es la media armónica de la precisión y el recall. Proporciona un buen equilibrio entre ambas métricas.

Fórmula:
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

### AUC-ROC (Área Bajo la Curva ROC):

Mide el rendimiento general de un modelo a través de todos los posibles umbrales de clasificación. Un AUC de 1 indica un clasificador perfecto, mientras que un AUC de 0.5 indica un clasificador aleatorio.

Fórmula:
El cálculo del AUC-ROC implica graficar la curva ROC (True Positive Rate vs. False Positive Rate) y calcular el área bajo esa curva.

### Priorizacion de metricas de evaluacion

El objetivo de este modelos de clasificacion es predecir correctamente cuando un cliente va a incumplir sus obligaciones, por ende, se prioriza la metrica Recall (sensividad) ya que el costo de un falso negativo es alto. 

# Resultados


## Conclusiones

## Autores

- Camilo Prada
- 
-
-

## Licencia

Este proyecto tiene la licencia MIT
