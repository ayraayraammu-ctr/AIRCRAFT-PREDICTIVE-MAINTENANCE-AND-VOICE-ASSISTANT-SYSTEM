# ✈️ Aircraft Predictive Maintenance System

## Overview

This project focuses on predicting the Remaining Useful Life (RUL) of aircraft engines using Machine Learning and Deep Learning techniques. By analyzing engine sensor data from the NASA FD001 dataset, the system can estimate the health of an engine and provide maintenance recommendations before failure occurs.

## Objectives

* Predict the Remaining Useful Life (RUL) of aircraft engines.
* Compare multiple Machine Learning algorithms.
* Implement a Deep Learning (LSTM) model for RUL prediction.
* Generate maintenance recommendations based on engine condition.
* Reduce unexpected failures and maintenance costs.

## Dataset

The project uses the NASA FD001 Turbofan Engine Degradation Simulation Dataset.

### Dataset Features

* Engine ID (`unit_nr`)
* Operating Cycles (`time_cycles`)
* Operational Settings (`op_setting_1`, `op_setting_2`, `op_setting_3`)
* Sensor Measurements (`sensor_1` to `sensor_21`)

### Target Variable

Remaining Useful Life (RUL)

RUL is calculated using:

RUL = Maximum Cycle − Current Cycle

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* TensorFlow / Keras
* Streamlit

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. RUL Calculation
4. Feature Selection
5. Data Scaling
6. Machine Learning Model Training
7. LSTM Model Development
8. Model Evaluation
9. Maintenance Recommendation Generation

## Machine Learning Models

### Linear Regression

Used as a baseline model for RUL prediction.

### Decision Tree Regressor

Uses tree-based decision rules to estimate RUL.

### Random Forest Regressor

Combines multiple decision trees to improve prediction accuracy.

### Gradient Boosting Regressor

Builds models sequentially to reduce prediction errors.

### XGBoost Regressor

An optimized boosting algorithm that achieved the best performance in this project.

## Deep Learning Model

### Long Short-Term Memory (LSTM)

LSTM is used because aircraft engine sensor readings are sequential time-series data.

Model Architecture:

* LSTM Layer (64 Units)
* Dropout Layer (0.2)
* LSTM Layer (32 Units)
* Dense Output Layer

## Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* R² Score

## Results

* Successfully predicted aircraft engine Remaining Useful Life.
* Compared multiple Machine Learning algorithms.
* XGBoost provided the highest prediction accuracy.
* LSTM effectively captured temporal patterns in sensor data.
* Generated maintenance recommendations based on predicted RUL.

## Maintenance Recommendation

| Predicted RUL | Recommendation                 |
| ------------- | ------------------------------ |
| Above 100     | Healthy Condition              |
| 50 – 100      | Routine Inspection Recommended |
| 20 – 50       | Maintenance Required Soon      |
| Below 20      | Immediate Maintenance Required |

## Future Enhancements

* Real-time sensor integration
* Cloud deployment
* Live aircraft monitoring
* Explainable AI techniques
* Real-time predictive analytics

## Conclusion

This project demonstrates the application of Machine Learning and Deep Learning techniques for predictive maintenance of aircraft engines. By accurately predicting Remaining Useful Life, the system helps improve safety, reduce maintenance costs, and prevent unexpected engine failures.


