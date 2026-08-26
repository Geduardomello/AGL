# Training & Evaluation Pipeline: Sinos River Basin (`AGL_Rio_Sinos-train_test_val.ipynb`)

Technical documentation and execution guide for the Jupyter Notebook implementing data preprocessing, temporal structuring, model training, and comparative evaluation (MSE vs. AGL Loss) for the **Sinos River Basin (RS, Brazil)**.

---

## 📌 1. Overview

This notebook implements the complete end-to-end river stage forecasting pipeline for the Sinos River targeting an **8-hour lead time horizon ($T+8\text{h}$)**. The workflow comprises:
1. **Sequence Engineering:** Tensor generation using historical sliding windows ($T-10$ to $T$) and projected future climate/discharge forcings ($T+8$ to $T+18$).
2. **Baseline Training (MSE):** Optimization of a Multi-Input BiLSTM architecture governed by standard symmetric Mean Squared Error.
3. **AGL Training (Innovation):** Training the identical dual-input architecture using the **Asymmetric Gradient Loss (AGL)**, integrating hydrograph wave kinetics and flood risk asymmetry.
4. **Hydrological & Lead-Time Evaluation:** Comparative assessment across global regression metrics ($R^2$, MAE, RMSE) and disaster-readiness operational indicators (peak capture rate, mean effective lead time, and extra anticipatory events).

---

## 🛠️ 2. Dependencies & Environment

* **Python Version:** `>= 3.10`
* **Core Libraries:**
  * `tensorflow >= 2.15.0`
  * `numpy >= 1.24.0`
  * `pandas >= 2.0.0`
  * `scikit-learn >= 1.3.0`
  * `plotly >= 5.18.0`
  * `matplotlib >= 3.8.0`
  * `joblib`

---

## 📥 3. Data Inputs

### 3.1 Base Dataset
* **File:** `dataset_engineered_enxuto.csv`
* **Description:** Hourly multivariate time-series combining in-situ telemetric river gauges with Open-Meteo atmospheric and land-surface reanalysis data.

### 3.2 Feature Set (14 Input Channels)
| Category | Variable Name | Description |
| :--- | :--- | :--- |
| **Target / Stage** | `nivel_cm_series` | River water level in centimeters at São Leopoldo gauge station |
| **Discharge** | `river_discharge_series_river_discharge (m³/s)_0`, `..._24` | Upstream and local river discharge dynamics ($m^3/s$) |
| **Atmospheric** | `rain_real`, `cloud_real`, `dew_real`, `humidity_real`, `temperature_real`, `weather_real`, `et0_real` | Hourly precipitation, cloud cover, dew point, relative humidity, air temperature, weather code, and reference evapotranspiration |
| **Soil Moisture** | `soil1_real`, `soil2_real`, `soil3_real`, `soil4_real` | Multi-layer volumetric soil moisture (0–7 cm, 7–28 cm, 28–100 cm, 100–255 cm) |

---

## ⚙️ 4. Temporal Structuring & Data Splitting

```python
janela = 10      # Historical input sequence length (10 time steps / hours)
horizonte = 8    # Target forecast horizon (8 hours ahead)
```

* **`X_passado` (Historical Branch):** Shape `(N, 10, 14)` — All 14 historical parameters.
* **`X_futuro` (Exogenous Forecast Branch):** Shape `(N, 10, 13)` — 13 future atmospheric and discharge features (river stage excluded).
* **`y` (Target):** River stage at $T+8\text{h}$.
* **Exported Preprocessed Array:** `lstm_duas_sequencias_h8.npz`

### Chronological Partitioning (Strict Sequential, No Shuffle):
* **Training Set:** $73\%$
* **Validation Set:** $18\%$
* **Test Set:** $9\%$

---

## 🧠 5. Neural Architecture (Multi-Input BiLSTM)

Both models (Baseline and AGL) share the exact same structural topology for fair ablation:
1. **Historical Context Branch:** `Input(shape=(10, 14))` $\rightarrow$ `BiLSTM(64 units)` $\rightarrow$ `Dropout(0.2)`
2. **Future Meteorological Branch:** `Input(shape=(10, 13))` $\rightarrow$ `LSTM(32 units)` $\rightarrow$ `Dropout(0.2)`
3. **Feature Fusion & Dense Head:** `Concatenate()` $\rightarrow$ `Dense(64, activation='relu')` $\rightarrow$ `Dense(32, activation='relu')` $\rightarrow$ `Dense(1)` (Linear Output)

---

## 🎯 6. AGL Hyperparameter Setup (Sinos Basin)

Parameters calibrated to account for the lowland floodplain response and high urban imperviousness of the Sinos River:
* **Flood Alert Threshold ($L$):** $350.0\,\text{cm}$ (Official municipal contingency trigger)
* **Noise Tolerance Threshold ($\theta$):** $2.0\,\text{cm}$
* **Rising Acceleration Weight ($W$):** $100.0$
* **Magnitude Multiplier ($\alpha$):** $1.0$
* **Dynamic Momentum Factor ($\beta$):** $100.0$
* **Underestimation Asymmetry Multiplier ($\gamma$):** $10000.0$

---

## 📈 7. Empirical Test Set Results

### 7.1 Regression & Operational Lead-Time Metrics ($T+8\text{h}$)
| Model | MAE (cm) | RMSE (cm) | $R^2$ | Mean Lead Time | Capture Rate (%) | Extra Events | Total Peaks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (MSE)** | $10.65$ | $14.87$ | $0.96$ | $1.33\,\text{h}$ | $100.00\%$ | $4$ | $21$ |
| **AGL (Custom)** | $41.62$ | $44.39$ | $0.63$ | **$11.43\,\text{h}$** | **$100.00\%$** | **$20$** | $21$ |

> **Operational Significance:** AGL trades symmetric baseflow precision for active flood anticipation, increasing the mean operational lead time from $1.33\,\text{h}$ to **$11.43\,\text{h}$** and boosting early warning issuances beyond the nominal 8-hour window from $4$ to **$20$**, securing indispensable evacuation margins for Civil Defense.

---

## 🚀 8. Execution Instructions

1. Ensure `dataset_engineered_enxuto.csv` is present in the working directory (or update the file path in Cell 2).
2. Execute the notebook cells sequentially:
   - **Cell 2:** Feature extraction, sliding window tensor generation, and export to `lstm_duas_sequencias_h8.npz`.
   - **Cell 4:** Training and evaluation of the **MSE Baseline** model.
   - **Cell 6:** Training and evaluation of the **AGL Loss** model.
   - **Cell 8:** Execution of interactive Plotly visualizer and computation of operational lead-time and peak-capture metrics.
