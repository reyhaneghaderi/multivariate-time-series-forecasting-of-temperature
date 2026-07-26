# Climate Forecasting with VAR & GRU
Short-term temperature forecasting (24-hour horizon) using classical statistical modeling (Vector Autoregression) versus deep learning (Gated Recurrent Unit) on the Jena Climate Dataset (2009–2016).

 - Project Overview
   Predicting weather is a classic time-series challenge. This project checks if a deep learning model (GRU) can beat a simple math model (VAR) at predicting air temperature 24 hours in advance using past weather data.

## Objectives

- Perform time-series preprocessing, resampling, and windowing on multi-year climate data.

- Benchmark a Vector Autoregression (VAR) model against a Gated Recurrent Unit (GRU) network.

- Evaluate both models on unseen test data using MAE and RMSE.

## Dataset & Features
   # Target Variable: 
       Air Temperature (T (degC))
       
   # Predictor Variables:   
    - Atmospheric pressure
    - Relative humidity
    -Wind speed
   - Historical temperature values
   -Two approaches were compared:

   # Sampling: 
    - Downsampled to 3-hour intervals to reduce high-frequency noise and computational overhead.

   # Methodology

  - Data Preprocessing
  -   Replaced invalid values (e.g., -9999) with NaN.
  -  Interpolated missing data and aligned time indices.
  -  Selected relevant meteorological features for forecasting.

   - Modeling
    - VAR: captured linear dependencies between multiple climate variables.
    - GRU: trained for 100 epochs on sequential windows of meteorological features.
   - Evaluation
   - Metrics: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

 - Compared forecasts against actual observed temperature values.  
