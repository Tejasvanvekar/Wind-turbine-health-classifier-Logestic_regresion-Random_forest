# Wind Turbine Health Classifier

A machine learning project for classifying wind turbine operational health status using **Logistic Regression** and **Random Forest** algorithms.

## Overview

This project uses time-series sensor data collected from wind turbines to classify their health status (e.g., Normal vs. fault conditions). The dataset contains hundreds of sensor readings captured at regular intervals, enabling predictive maintenance.

## Dataset

The project includes two CSV datasets (`wind_turbine_snippet_A.csv` and `wind_turbine_snippet_B.csv`) stored using Git LFS. Each file contains:

- `time_stamp` – Timestamp of the measurement
- `asset_id` – Unique identifier for each turbine
- `status` – Health label (e.g., `Normal`, fault types)
- 200+ sensor features including:
  - Generator metrics (acceleration, speed, current, voltage)
  - Temperature readings (bearings, gearbox, stator windings, oil)
  - Hydraulic pressure measurements
  - Vibration sensors (nacelle longitudinal/transverse)
  - Wind speed and direction
  - Power output (active/reactive power)

## Models

### Logistic Regression
A linear classification model used as a baseline to distinguish between healthy and faulty turbine states.

### Random Forest
An ensemble tree-based model providing higher accuracy and feature importance insights for turbine fault detection.

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- Git LFS (for large CSV files)

## Usage

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib
```

Load and preprocess the data using `pandas`, then train and evaluate the classifiers with `scikit-learn`.

## Results

Both models are evaluated using standard classification metrics:
- Accuracy
- Precision / Recall
- F1-Score
- Confusion Matrix
