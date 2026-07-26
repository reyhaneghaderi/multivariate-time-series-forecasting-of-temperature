# Climate Forecasting: VAR vs. GRU

Short-term air-temperature forecasting at a **24-hour horizon** using a classical statistical model, **Vector Autoregression (VAR)**, and a deep-learning model, **Gated Recurrent Unit (GRU)**, on the **Jena Climate Dataset (2009–2016)**.

## Project Overview

Predicting meteorological variables is a multivariate time-series forecasting problem. This project compares a recurrent neural network with a classical statistical benchmark for predicting air temperature 24 hours ahead from historical climate measurements.

Both models are evaluated on unseen test data using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

## Objectives

- Clean and prepare multi-year meteorological time-series data.
- Downsample the original 10-minute observations to 3-hour intervals.
- Create historical input windows for forecasting.
- Train a Vector Autoregression model as a statistical baseline.
- Train a Gated Recurrent Unit network for nonlinear forecasting.
- Compare both models using the same 24-hour prediction horizon.
- Evaluate model performance on unseen test data.

## Dataset

This project uses the **Jena Climate Dataset**, recorded by the Max Planck Institute for Biogeochemistry in Jena, Germany.

- **Period:** 2009–2016
- **Original sampling frequency:** Every 10 minutes
- **Working sampling frequency:** Every 3 hours
- **Target variable:** Air temperature, `T (degC)`
- **Forecast horizon:** 24 hours

### Dataset download

Download the original dataset from TensorFlow:

[Download the Jena Climate Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)

A ZIP copy is also stored in this repository at:

```text
data/raw/jena_climate_2009_2016.zip
```

After downloading or cloning the repository, extract the ZIP file and place the CSV file at:

```text
data/raw/jena_climate_2009_2016.csv
```

## Selected Features

The models use the following meteorological variables:

- Atmospheric pressure: `p (mbar)`
- Air temperature: `T (degC)`
- Relative humidity: `rh (%)`
- Wind speed: `wv (m/s)`

Historical temperature is included as an input because past temperature values contain strong temporal information for short-term forecasting.

## Preprocessing Pipeline

### 1. Data cleaning

Invalid sentinel values such as `-9999` were identified and replaced with missing values.

### 2. Missing-value treatment

Linear interpolation was applied to fill missing observations and maintain a continuous datetime index.

### 3. Downsampling

The original 10-minute observations were reduced to 3-hour intervals to decrease computational cost and reduce high-frequency variation.

### 4. Feature selection

Redundant and highly correlated variables were removed to reduce unnecessary model complexity and multicollinearity.

### 5. Scaling

Numerical features were standardised using `StandardScaler`.

### 6. Sequence construction

The GRU input data were converted into three-dimensional tensors:

```text
(samples, time steps, features)
```

The final GRU input configuration was:

```text
Time steps: 56
Features: 4
Sliding step: 1
```

Because each observation represents 3 hours:

```text
56 × 3 hours = 168 hours = 7 days
```

Each GRU input therefore contains seven days of historical measurements.

## Methodology

### Vector Autoregression

VAR is a classical multivariate statistical model in which all included variables are treated as endogenous.

- **Lag-selection criterion:** Akaike Information Criterion
- **Selected lag:** 24
- **Historical period represented by lag 24:** 72 hours
- **Forecasting strategy:** Recursive 8-step prediction
- **Forecast horizon:** 8 × 3 hours = 24 hours

### Gated Recurrent Unit

The GRU model was designed to learn nonlinear temporal dependencies from multivariate climate sequences.

#### Architecture

- Two stacked GRU layers
- 128 units per GRU layer
- Recurrent dropout: 0.3
- Dense layers after the recurrent layers
- Final output layer for temperature prediction

#### Training

- Optimiser: Adam
- Learning rate: 0.001
- Loss function: MAE
- Maximum epochs: 100

## Results

| Model | Model type | MAE (°C) | RMSE (°C) | Result |
|---|---|---:|---:|---|
| VAR, lag 24 | Statistical multivariate model | **2.438** | **3.100** | Best test performance |
| GRU, 128 units | Deep recurrent neural network | 4.107 | 5.132 | Overfitting after the best validation epoch |

## Key Findings

### VAR achieved the strongest test performance

VAR produced lower MAE and RMSE than the final GRU model. This suggests that short-term temperature variation in this experiment contained strong linear autoregressive structure.

### The GRU model overfit

The GRU reached its best validation performance at approximately epoch 21, with validation MAE around `2.55 °C`. Continuing training to 100 epochs reduced its ability to generalise to the test set.

This result does not prove that GRU models are generally worse than VAR. It shows that, under this architecture and training procedure, the GRU model was not stopped at its best validation point.

### Early stopping was necessary

The final GRU result demonstrates the importance of:

- early stopping;
- restoring the best model weights;
- tuning dropout and recurrent dropout;
- reducing model size when necessary;
- evaluating the best validation checkpoint instead of the final epoch.

## Visual Results

### Meteorological variables over time

![Meteorological variables over time](./results/figures/meteorological_variables_time_series.png)

### Correlation matrix

![Correlation matrix](./results/figures/correlation_matrix.png)

### GRU prediction versus actual temperature

![GRU prediction versus actual temperature](./results/figures/gru_prediction_vs_actual.png)

### VAR 24-hour prediction versus actual temperature

![VAR prediction versus actual temperature](./results/figures/var_24h_prediction_vs_actual.png)

### GRU and VAR comparison

![GRU and VAR comparison](./results/figures/gru_var_vs_actual_24h_forecast.png)

## Repository Structure

```text
multivariate-time-series-forecasting-of-temperature/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       ├── .gitkeep
│       └── jena_climate_2009_2016.zip
│
├── notebooks/
│   └── temperature_forecasting.ipynb
│
└── results/
    └── figures/
        ├── meteorological_variables_time_series.png
        ├── correlation_matrix.png
        ├── gru_prediction_vs_actual.png
        ├── var_24h_prediction_vs_actual.png
        └── gru_var_vs_actual_24h_forecast.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/reyhaneghaderi/multivariate-time-series-forecasting-of-temperature.git
cd multivariate-time-series-forecasting-of-temperature
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Extract:

```text
data/raw/jena_climate_2009_2016.zip
```

The extracted CSV should be located at:

```text
data/raw/jena_climate_2009_2016.csv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebook inside the `notebooks/` folder.

## Requirements

The project uses:

- Python
- pandas
- NumPy
- TensorFlow
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
- Jupyter

## Limitations

- The GRU was trained for 100 epochs without restoring the best validation checkpoint.
- Only one GRU architecture was evaluated.
- Hyperparameter optimisation was limited.
- The comparison is based on one dataset and one forecast horizon.
- Recursive VAR forecasting may accumulate errors across multiple steps.
- Strong seasonal patterns may favour linear autoregressive models in this configuration.

## Future Improvements

- Add `EarlyStopping` with `restore_best_weights=True`.
- Add model checkpointing.
- Tune GRU size, dropout, sequence length and learning rate.
- Compare GRU with LSTM, Temporal Convolutional Networks and Transformer-based models.
- Add persistence and seasonal-naive baselines.
- Evaluate multiple forecast horizons.
- Report training time and computational cost.
- Perform rolling-origin cross-validation.
- Add uncertainty estimates or prediction intervals.

## Conclusion

The VAR model achieved the best test performance, with an MAE of `2.438 °C` and an RMSE of `3.100 °C`. The GRU model achieved an MAE of `4.107 °C` and an RMSE of `5.132 °C`.

The experiment shows that a more complex neural network does not automatically produce better forecasts. Careful validation, regularisation and checkpoint selection are essential. For this dataset and 24-hour forecast horizon, the VAR model provided the stronger and more reliable result.
