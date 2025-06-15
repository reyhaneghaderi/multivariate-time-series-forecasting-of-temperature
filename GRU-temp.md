
# Multivariate Time Series Forecasting of Temperature using GRU

This project applies a GRU-based deep learning model to forecast temperature using historical weather data from the Jena Climate dataset. 
## 📁 Dataset
- Source: [Jena Climate Dataset]
- Features include: Pressure, humidity, wind speed, solar radiation, etc.
- Temporal resolution: Resampled to every **3 hours**

## 📈 Methodology

✅ Preprocessed multivariate time series data  
✅ Normalized using `StandardScaler`  
✅ Created sliding windows (56 timesteps) with a step of 2  
✅ Forecasted temperature 7 hours ahead as the target  
✅ Split data into train (8000), validation (1500), and test (remaining)  
✅ Built a 2-layer GRU model with dropout and dense layers  
✅ Recursive prediction for 24 future time steps  
✅ Visualized the results

## 🧠 Model Architecture

- `GRU(128, return_sequences=True)`
- `GRU(128)`
- `Dense(128, ReLU)`
- `Dense(1, Linear)`

## 🔁 Recursive Forecasting

Starting from the last test window, the model forecasts 1 hour ahead. This prediction is appended to the input window and the oldest timestep is dropped. The process is repeated for 24 steps.

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- NumPy / pandas
- Matplotlib

