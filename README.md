# Credit Card Default Risk - Riesgo de Incumplimiento de Tarjetas de Credito

El riesgo de incumplimiento de tarjetas de crédito es un desafío recurrente en la industria financiera. Este fenómeno se refiere a la probabilidad de que un titular de tarjeta de crédito sea incapaz o se niegue a pagar el saldo total o mínimo de su cuenta dentro del plazo establecido.

Este proyecto tiene como objetivo desarrollar distintos modelos de machine learning de clasificacion capaz de predecir con precisión la probabilidad de que un cliente incurra en un incumplimiento de pago. Para lograr esto, se llevará a cabo un análisis exhaustivo de un conjunto de datos históricos de clientes de tarjetas de crédito proporcionado por American Express para el concurso AmExpert CodeLab 2021, que puede encontrarse en kaggle siguiendo este [enlace](https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021).

Predecir el riesgo de incumplimiento de tarjetas de crédito es crucial para las empresas de crédito por varias razones:

* Gestión de Riesgos: Permite a las instituciones financieras identificar a los clientes con mayor probabilidad de incumplimiento, lo que facilita la toma de decisiones informadas sobre la concesión de crédito y la gestión de carteras.
* Prevención de Pérdidas: Al identificar a los clientes de alto riesgo, las empresas pueden implementar medidas preventivas, como ajustar los límites de crédito o contactar a los clientes para ofrecer soluciones.
* Mejora de la Rentabilidad: Al reducir las pérdidas por incumplimiento, las empresas pueden mejorar su rentabilidad y fortalecer su posición en el mercado.
* Desarrollo de Productos Personalizados: Los modelos de predicción pueden utilizarse para desarrollar productos y servicios financieros personalizados que se adapten a las necesidades y riesgos específicos de cada cliente.

# Credit Card Default Risk

Credit card default risk is a recurring challenge in the financial industry. This phenomenon refers to the likelihood that a credit card holder will be unable or unwilling to pay the total or minimum balance of their account within the established timeframe.

This project aims to develop various machine learning classification models capable of accurately predicting the probability that a customer will default on a payment. To achieve this, a comprehensive analysis of a historical dataset of credit card customers provided by American Express for the AmExpert CodeLab 2021 competition will be conducted, which can be found on Kaggle by following this [link](https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021).
 

Predicting credit card default risk is crucial for credit companies for several reasons:

* Risk Management: It allows financial institutions to identify customers with a higher likelihood of default, facilitating informed decision-making regarding credit granting and portfolio management.

* Loss Prevention: By identifying high-risk customers, companies can implement preventive measures, such as adjusting credit limits or contacting customers to offer solutions.

* Profitability Improvement: By reducing losses due to default, companies can improve their profitability and strengthen their market position.

* Development of Customized Products: Predictive models can be used to develop personalized financial products and services tailored to the specific needs and risks of each customer.

# Metodologia

El proyecto se divide en los siguientes pasos:

1. Análisis exploratorio de los datos
2. Preprocesamiento de los datos
3. Optimización de hiperparámetros.
4. Entrenamiento de los modelos
5. Evaluación de los modelos
5. Comparacion de los modelos de clasificacion
9. Conclusiones

# Methodology

The project is divided into the following steps:

1. Exploratory Data Analysis
2. Data Preprocessing
3. Hyperparameter Optimization
4. Model Training
5. Model Evaluation
6. Comparison of Classification Models
7. Conclusions

# Analisis Exploratorio de Datos (EDA) Dataset AmExpert CodeLab 2021

Puede ver el Notebook del EDA en este [enlace](https://github.com/LiceoM/Credit_Risk_ML/blob/main/Code/EDA.ipynb).

You can view the EDA Notebook at this [link](https://github.com/LiceoM/Credit_Risk_ML/blob/main/Code/EDA.ipynb).

La metodologia de analisis exploratorio de datos consiste en los siguientes pasos:

* Imputacion valores Nulos
* Eliminacion de outliers
* Transformacion de variables categoricas con LabelEncoder()
* Escalamiento de variables continuas con StandarScaler()

# Exploratory Data Analysis (EDA) for AmExpert CodeLab 2021 Dataset

The methodology for exploratory data analysis consists of the following steps:

*Imputation of Missing Values
* Elimination of Outliers
* Transformation of Categorical Variables with LabelEncoder()
*Scaling of Continuous Variables with StandardScaler()

## Modelos de Machine Learning

Se probaron diferentes modelos de machine learning para determinar cual es el mejor modelo para este problema, los modelos que se probaron son los siguientes:

- Logistic Regression
- KNN
- Random Forest
- XgBoost

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

### Prioritization of Evaluation Metrics
The goal of these classification models is to accurately predict when a customer will default on their obligations. Therefore, the Recall (sensitivity) metric is prioritized since the cost of a false negative is high.

# Resultados

Despues de aplicar cuatro modelos de machine learning para el problema de clasificacion, se obtuvieron los mejores resultados con el modelo de Xboost. 

### Results
After applying four machine learning models to the classification problem, the best results were obtained with the XGBoost model

Puedes ver los resultados en el siguiente enlace: [Modelos de Clasificacion](https://github.com/LiceoM/Credit_Risk_ML/blob/main/Code/Classification_models.ipynb)

You can see the results at the following link: [Classification Models](https://github.com/LiceoM/Credit_Risk_ML/blob/main/Code/Classification_models.ipynb)


### Resultados de los modelos

### Model Performance Comparison

| Modelo              | Accuracy | Precision | Recall   | F1 Score | AUC      |
|---------------------|----------|-----------|----------|----------|----------|
| KNN                 | 0.941814 | 0.622089  | 0.854489 | 0.720000 | 0.902341 |
| `XGBoost`             | 0.951753 | 0.645291  | `0.996904` | 0.783455 | 0.972162 |
| Regresion Logistica | 0.950669 | 0.646773  | 0.961816 | 0.773444 | 0.955708 |
| Random Forest       | 0.955186 | 0.665500  | 0.981424 | 0.793161 | 0.967046 |


### Matriz de Confusion del mejor modelo: Xgboost

| Actual \ Predicted | Prediccion Positiva   | Prediccion Negativa|
|-------------------|------|------|
| Clase Negativa    | 9568 | 531  |
| Clase Positiva    | 3    | 966  |

El modelo prefijo correctamente 966 de las 969 clases positivas del dataset de prueba. 

The Xgboost model predicted 966 out of 969 Positive Classes in the testing dataset.

### Metricas del mejos modelo: Xgboost

* Accuracy: 0.9518
* Precision: 0.6453
* Recall: 0.9969
* F1 Score: 0.7835
* AUC: 0.9944

## Conclusiones

El mejor modelo para este problema de clasificacion fue el Xgboost, la gran capacidad que ofrece el modelo para encontrar los hiperparametros optimos que arrojan el mejor puntaje de recall lo hace un modelo atractivo para resolver problemas de clasificacion.

The best model for this classification problem was XGBoost. The model's great capability to find the optimal hyperparameters that yield the best recall score makes it an attractive option for solving classification problems.

## Autores

- Camilo Prada [LinkedIn](https://www.linkedin.com/in/andres-prada-4051a2250/)

## Licencia

Este proyecto tiene la licencia MIT
