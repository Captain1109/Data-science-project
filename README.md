# Diabetes Risk Prediction - Machine Learning Project
This project utilizes the **BFRSS 2015 Diabetes Indicators dataset** to build a machine learning model that predicts the risk of diabetes based on various health indicators. The model was developed using multiple machine learning algorithms and evaluated for performance using various metrics such as accuracy, precision, recall, and F1-score.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Project Overview

This machine learning project involves analyzing and predicting the risk of diabetes. The dataset contains various health-related features like age, BMI, blood pressure, exercise habits, and others. The goal is to predict the likelihood of an individual having diabetes, which is represented as a binary classification problem (1 = Diabetes, 0 = No Diabetes).

The project demonstrates the use of data preprocessing, model selection, and evaluation using key machine learning algorithms, including logistic regression, extreme gradient boost, multilayer perceptron, and support vector machine.

## Dataset

The dataset used for this project is sourced from kaggle. It is an updated version of the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey dataset. It contains 70692 rows and 22 columns, with the following columns:

- **Age**
- **BMI (Body Mass Index)**
- **Blood Pressure**
- **Physical Activity**
- **Physical health**
- **Cholesterol level**
- **Diabetes (Target)**: 1 for diabetes, 0 for non-diabetes

You can access the dataset here: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset].

## Installation Instructions

To run this project on your local machine, ensure that you have the following libraries installed. You can install them using `pip`:


```bash
pip install -r requirements.txt
```

## Data Preprocessing
The dataset was preprocessed using the following steps:

    Handling Missing Values: Rows with missing or null values were dropped or imputed with the mean/median where appropriate.

    Feature Scaling: Numerical features such as age, BMI, and blood pressure were normalized using MinMaxScaler to ensure uniform scale during model training.

    Feature Encoding: Categorical features such as physical activity and family history were encoded using LabelEncoder or OneHotEncoder.

    Splitting Data: The dataset was split into training and testing sets (80% train, 20% test).

## Model Development
The following machine learning models were implemented to predict the risk of diabetes:

    Logistic Regression: Used as a baseline model for binary classification.
    XGBoost: Widely used in medical diagnosis systems, such as predicting the likelihood of diabetes or heart disease based on patient features.
    MLP (Multilayer Perceptron): Commonly applied in classification tasks.
    Support Vector Machine (SVM): A powerful widely used algorithm for classification.

Each model was trained using a standard 80/20 split and evaluated on the test set.

## Model Evaluation
The evaluation method used include accuracy, correlation matrix, precision, recall and f1-score.

## Results
The logistic regression model had:
    Accuracy: 75.1%
    Precision: 75%
    Recall: 75%
    F1-Score: 75%
The XGBoost model had:
    Accuracy: 75.3%
    Precision: 75%
    Recall: 75%
    F1-Score: 75%
The MLP model had:
    Accuracy: 75.1%
    Precision: 75%
    Recall: 75%
    F1-Score: 75%
The SVM model had:
    Accuracy: 75.1%
    Precision: 76%
    Recall: 76%
    F1-Score: 76%