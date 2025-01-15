# Neural Network Regression Model

This repository contains a series of models developed for a regression task, including traditional machine learning models like **Linear Regression** and **CatBoostRegressor**, as well as an **Artificial Neural Network (ANN)** implemented using TensorFlow.

## Overview:

Initially, the project started with traditional machine learning techniques (Linear Regression and CatBoost) to solve the regression problem. Later, the model complexity was increased by transitioning to an Artificial Neural Network (ANN) built with TensorFlow for improved performance and flexibility.

The goal of this project is to predict a continuous target variable based on multiple input features. The dataset has been preprocessed with normalization and whitening techniques to ensure better performance of the models.

## Requirements:

To run this project, make sure to install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Dependencies:
- pandas
- numpy
- scikit-learn
- scipy
- tensorflow
- catboost

## Dataset:
The dataset consists of multiple input features that are used to predict a continuous target variable. The dataset is split into training and test sets, and preprocessing steps like **normalization** and **whitening** are applied to the features before training the models.
the dataset is from [ShrutiiBhosale](https://github.com/ShrutiiBhosale/Real_Estate_Price_Prediction).

## Models:

1. Linear Regression:
The first model developed was a simple linear regression model using `sklearn.linear_model.LinearRegression`. This model assumes a linear relationship between the input features and the target variable.

2. CatBoost Regressor:
The second model used is `CatBoostRegressor`, a powerful gradient boosting algorithm. CatBoost is known for its efficiency with categorical variables and its robustness in many regression tasks.

3. Artificial Neural Network (ANN):
Finally, a neural network was implemented using TensorFlow (`tf.keras`). This model uses multiple layers with **ReLU** activation functions to capture complex patterns in the data. The model was trained using the **Adam optimizer** with a learning rate of 0.0001 and evaluated on the test set.

## Model Evaluation:

The models were evaluated based on their performance on a separate test set. For the ANN model, the **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** metrics were used to assess the quality of the predictions. For linear and CatBoost models, **R-squared** and **MAE** were used for evaluation.

## Evaluation Results:

The final evaluation results include:

- **Test Loss (MSE)**: The mean squared error of the model's predictions on the test set.
- **Test MAE**: The mean absolute error of the predictions, showing the average magnitude of errors.
- **Test Accuracy**: (Not typically used in regression tasks but included for completeness).

