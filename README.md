# Beijing_Air-Quality_Prediction

## Introduction
Welcome to the Beijing Air Quality Forecasting Project, where we tackle the challenge of predicting PM2.5 concentrations in one of the world's most polluted cities. This project leverages time series analysis and deep learning, specifically Long Short-Term Memory (LSTM) networks, to provide insights into future air quality based on historical environmental data. Understanding and predicting pollution levels is crucial for public health, urban planning, and environmental policy.

Time Series Forecasting in Air Quality Prediction
Predicting air quality involves looking at past data to forecast future PM2.5 levels. This is important because it helps in making informed decisions about public health measures, pollution control strategies, and personal activities. The complexity of air pollution, influenced by factors like weather, traffic, and industrial emissions, makes forecasting both vital and challenging.

## Dataset Overview
Our dataset spans from January 2010 to 2013, providing hourly measurements of various environmental factors like dew point, temperature, pressure, wind speed, and PM2.5 concentration from multiple stations in Beijing.

## Data Preparation
Before diving into modeling, I prepared our data with the following steps:

- Data Cleaning: Missing values were addressed by filling them with the mean of each respective column, maintaining the consistency of our time series.
- Normalization: We normalized our features using StandardScaler to ensure all variables contributed equally to the model, avoiding any bias from different scales.

```bash
from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
```

These steps were crucial to make our LSTM model work efficiently with the data.

## Model Architecture
We built a Bidirectional LSTM model to predict PM2.5 levels, which is adept at capturing the temporal dynamics from both past and future perspectives, enhancing our ability to understand the flow of pollution over time.

Model Summary:

- Bidirectional LSTM layers: These layers look at the sequence in both directions, which is perfect for understanding the cyclical nature of air pollution.
- BatchNormalization: This helps in stabilizing the learning process by normalizing the inputs to each layer.
- A Dense layer

```bash 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization

# Building the model
model = Sequential([
        Bidirectional(LSTM(50, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        BatchNormalization(),
        Bidirectional(LSTM(25, activation='tanh', kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dense(1)
    ])
```
This architecture was chosen because of its effectiveness in handling the sequential nature of our air quality data.

## Experiments
I did a totla of 16 experiments and submitted 15 of them on kaggle, with the best one having a RMSE of **61.91** , the experimentation process comprised of many parameter tuning, feature engineering and Model changes, all this is well documented in my report and the notebook.

## Results
Our model's performance was measured using the Root Mean Squared Error (RMSE). Here's how we calculated it:

```bash
# Evaluate model performance
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
```
The model's performance was evaluated using the Root Mean Squared Error (RMSE). Below are graphs:

1. Comparing the predicted and actual values
![image](https://github.com/user-attachments/assets/862d2532-2bde-4aaf-acff-40d0e0cf1e16)

2. Showing the Final training loss against the loss on epochs
![image](https://github.com/user-attachments/assets/8afe1b76-891c-4141-8c1d-a25c01040a79)

The model showed good capability in predicting general trends in PM2.5 concentrations, although capturing sudden changes proved more difficult due to the volatile nature of pollution sources.

## Conclusion
Working on this project was enlightening, showing how complex air quality forecasting can be. The LSTM model, after refinement through multiple experiments, was able to provide useful predictions for long-term trends in PM2.5 levels. The unpredictability of short-term pollution spikes remains a challenge. Future work could explore integrating additional data sources or experimenting with alternative model structures to enhance prediction accuracy. Remember, this project not only aids in understanding air quality but also contributes to broader environmental health initiatives.
