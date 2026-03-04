# Wind Turbine Anomaly Detection

## Overview
This repository contains a comprehensive data science notebook focused on detecting anomalous behavior in wind turbines using sensor data. The primary objective is to build a machine learning classification model capable of distinguishing between normal and anomalous operational states, and subsequently predicting the status of unknown observations.

## Problem Statement
The dataset consists of measurements from two wind turbines (Turbine 38 and Turbine 44) equipped with ~238 sensors recording various physical quantities (such as temperatures, pressures, wind speeds, and power outputs) every 10 minutes. Each observation is labeled with a health status:
- **Normal**: The turbine is operating correctly.
- **Anomalous**: The turbine behavior deviates from normal operation (potential fault or degradation).
- **Unknown**: The operational status has not been determined.

The goal is to accurately classify the records despite the severe class imbalance (where normal operation dominates, and anomalies are rare).

## Project Workflow
1. **Data Exploration & Preprocessing**:
   - Analyzed the temporal distribution to see if anomalies are clustered or scattered.
   - Validated Active Power values and plotted Power Curves (Wind Speed vs. Active Power) to visualize turbine operating regimes.
   - Scaled numerical features using `StandardScaler` to prepare for model ingestion.

2. **Model Selection & Training**:
   - **Logistic Regression (Baseline)**: A linear baseline model configured with L1 penalty for feature selection, and customized `class_weight={0: 1, 1: 3}` to counteract the severe imbalance between normal and anomalous classes.

3. **Appropriate Evaluation Metrics**:
   - **Macro F1-Score**: Chosen over Accuracy because relying on Accuracy in heavily imbalanced datasets leads to the "Accuracy Paradox."
   - **Precision-Recall AUC (PR-AUC)**: Chosen over ROC-AUC because it strictly evaluates performance on the minority class rather than rewarding the model for identifying easily recognizable normal states.
   - **Log Loss**: Measures the confidence of the predicted probabilities.
   - **Confusion Matrix**: Translated the errors into business impact (False Alarms vs. Missed Faults).

4. **Predicting the Unknowns**:
   - Scaled the unknown data using the scaler fitted on historical training data.
   - Retrained the best model configuration on 100% of the labeled data to maximize knowledge extraction.
   - Performed out-of-sample predictions on the unknown dataset (winter data) to estimate global anomaly rates.

## Key Files
* `FINAL.ipynb`: The main Jupyter Notebook containing the full Data Science workflow, data exploration, visualizations, and modeling results.
* `wind_turbine_snippet_A.csv` & `wind_turbine_snippet_B.csv`: The datasets utilized for model training and prediction.

## Requirements
To run this project, you will need Jupyter Notebook and the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
