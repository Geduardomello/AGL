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
  

   # Leave-One-Event-Out (LOEO) Flood Generalization: Sinos River 2024 Catastrophic Event

Technical documentation and execution guide for `AGL_Rio_Sinos_LOEO_2.ipynb`, evaluating out-of-distribution generalization, peak level extrapolation, and the **Peak Smoothing Paradox** during the **May 2024 Catastrophic Flood Event** ($811.00\text{ cm}$) in the Sinos River Basin.

---

## 📌 1. Overview

Hydrological machine learning models routinely succumb to the **Peak Smoothing Paradox**: because extreme flood peaks constitute an extreme statistical minority in training records (<13%), symmetric objective functions like Mean Squared Error (MSE) optimize predominantly for baseflow conditions, drastically underestimating maximum flood amplitudes.

This notebook implements a strict **Leave-One-Event-Out (LOEO)** experimental protocol:
* **Catastrophic Event Isolation:** The unprecedented May 2024 flood surge (reaching an all-time crest of $811.00\text{ cm}$ at index 50376) is completely removed from the training and validation sets.
* **Out-of-Distribution Stress Test:** Models are trained solely on moderate events and normal river regimes, then challenged to extrapolate zero-shot dynamics during the catastrophic wave.
* **Controlled Ablation:** Identical dual-input BiLSTM architectures trained with **MSE Loss** versus **Asymmetric Gradient Loss (AGL)**.

---

## 🛠️ 2. Dependencies & Environment

* **Python:** `>= 3.10`
* **Core Libraries:**
  ```text
  tensorflow>=2.15.0
  numpy>=1.24.0
  pandas>=2.0.0
  scikit-learn>=1.3.0
  plotly>=5.18.0
  matplotlib>=3.8.0
  ```

---

## 📥 3. Data Inputs & LOEO Isolation Scheme

### 3.1 Input Tensor
* **Source File:** `lstm_duas_sequencias_h8.npz`
  * `X_passado`: Shape `(60267, 10, 14)` — Sliding historical window ($T-10$ to $T$, 14 channels including river stage, upstream/local discharge, precipitation, soil moisture, and atmospheric variables).
  * `X_futuro`: Shape `(60267, 10, 13)` — Exogenous forecast window ($T+8$ to $T+18$, 13 weather and runoff channels; river stage excluded).
  * `y`: Shape `(60267, 1)` — Ground truth river water level ($	ext{cm}$) at $T+8\text{h}$.

### 3.2 Splitting Logic
```python
# 1. Absolute Peak Detection
indice_pico = np.argmax(y)  # Index 50376 (811.00 cm)

# 2. Window Isolation (2,700 hourly steps ≈ 112.5 days around the event)
margem_antes, margem_depois = 1500, 1200
inicio_test = indice_pico - margem_antes  # Index 48876
fim_test = indice_pico + margem_depois    # Index 51576

# 3. Test vs. Train/Val Partitions
X1_test, X2_test, y_test = X_passado[inicio_test:fim_test], X_futuro[inicio_test:fim_test], y[inicio_test:fim_test]
indices_resto = np.setdiff1d(np.arange(len(y)), np.arange(inicio_test, fim_test))

# 80/20 Chronological Split on Remaining Records
n_treino = int(len(indices_resto) * 0.8)
X1_train, X2_train, y_train = X_passado[indices_resto][:n_treino], X_futuro[indices_resto][:n_treino], y[indices_resto][:n_treino]
X1_val, X2_val, y_val = X_passado[indices_resto][n_treino:], X_futuro[indices_resto][n_treino:], y[indices_resto][n_treino:]
```

---

## 🧠 4. Model Topology

Both benchmark models share the exact same structural network:
* **Historical Stream:** `Input(shape=(10, 14))` $\rightarrow$ `LSTM(64 units)` $\rightarrow$ `Dropout(0.2)`
* **Future Exogenous Stream:** `Input(shape=(10, 13))` $\rightarrow$ `LSTM(32 units)` $\rightarrow$ `Dropout(0.2)`
* **Head:** `Concatenate()` $\rightarrow$ `Dense(64, activation='relu')` $\rightarrow$ `Dense(32, activation='relu')` $\rightarrow$ `Dense(1, activation='linear')` (Unconstrained linear output enabling positive extrapolation above the historical normalized maximum).
* **Optimization:** `Adam(learning_rate=0.001)`, `batch_size=32`, `epochs=100`.

---

## 🎯 5. AGL Loss Configuration

Calibrated for the low-gradient floodplain dynamics of São Leopoldo:
* **Alert Threshold ($L$):** $350.0\text{ cm}$
* **Velocity Noise Tolerance ($\theta$):** $2.0\text{ cm}$
* **Rising Acceleration Weight ($W$):** $100.0$
* **Magnitude Multiplier ($\alpha$):** $1.0$
* **Dynamic Wave Momentum ($\beta$):** $100.0$
* **Underestimation Risk Factor ($\gamma$):** $10000.0$

```python
def flood_weighted_loss_momento(alpha=1.0, beta=100.0, gamma=10000.0, L=L_scaled, theta=theta_scaled, W=100.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        dy = tf.concat([[0.0], y_true[1:] - y_true[:-1]], axis=0)
        
        wi = 1.0 + W * tf.cast(dy > theta, tf.float32)
        aceleracao_term = beta * tf.nn.relu(dy) * tf.square(y_true - y_pred)
        base_term = 1.0 + alpha * tf.nn.relu(y_true - L)
        mse_term = tf.square(y_true - y_pred)
        under_term = gamma * tf.cast(y_true > L, tf.float32) * tf.square(tf.nn.relu(y_true - y_pred))

        numerador = tf.reduce_sum(wi * (base_term * mse_term + under_term + aceleracao_term))
        denominador = tf.reduce_sum(wi) + 1e-7
        return numerador / denominador
    return loss
```

---

## 📈 6. Experimental Validation (May 2024 Event: $811.00\text{ cm}$)

### 6.1 Performance Comparison on Isolated Unseen Peak ($T+8\text{h}$)
| Model Objective | MAE ($	ext{cm}$) | $R^2$ Score | Observed Crest | Predicted Crest | Absolute Peak Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (MSE Loss)** | $26.80$ | $0.94$ | $811.00\text{ cm}$ | $670.60\text{ cm}$ | **$-140.40\text{ cm}$ (Severe Underestimation)** |
| **AGL Loss (Ours)** | **$11.38$** | **$0.99$** | $811.00\text{ cm}$ | **$791.90\text{ cm}$** | **$-19.10\text{ cm}$** |

```
    Peak Elevation Comparison (May 2024 Crest: 811 cm)
    --------------------------------------------------
    Observed Crest    : [========================================] 811.0 cm
    AGL Loss (Ours)   : [======================================  ] 791.9 cm (-19.1 cm)
    Baseline (MSE)    : [=================================       ] 670.6 cm (-140.4 cm)
```

---

## 🚀 7. Execution Guide

1. Place `lstm_duas_sequencias_h8.npz` in your notebook root directory.
2. Run cells sequentially:
   - **Cell 1 (`Code`):** Execute LOEO event isolation, MinMax normalization, and train the **AGL Loss BiLSTM** model.
   - **Cell 2 (`Code`):** Train the **MSE Baseline BiLSTM** model on identical data slices.
   - **Cell 3 (`Code`):** Generate interactive Plotly hydrographs comparing ground truth against both predicted trajectories.
