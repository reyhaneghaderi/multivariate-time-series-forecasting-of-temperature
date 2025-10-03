# Climate Forecasting with VAR & GRU
 - Project Overview
    - This project focuses on short-term climate forecasting using the Jena Climate dataset (2009–2016).
    - The main task was to predict air temperature 24 hours ahead using multiple meteorological features:
    - Atmospheric pressure
    - Relative humidity
    -Wind speed
   - Historical temperature values
   -Two approaches were compared:
    -VAR (Vector AutoRegression) – a classical statistical model for multivariate time series.
   - GRU (Gated Recurrent Unit) – a deep learning recurrent neural network.

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
