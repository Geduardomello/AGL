Asymmetric Gradient Loss (AGL) for Flood Forecasting
Official repository for the framework and experimental validation of Asymmetric Gradient Loss (AGL): Integrating Hydrological Momentum into Deep Learning for Flood Prediction.
Overview
Hydrological time-series forecasting during flash floods and extreme flood events suffers from two structural pathologies when trained with standard objective functions (e.g., Mean Squared Error - MSE):
Predictive Inertia (Phase Lag): Models fail to capture rapid wave acceleration, lagging hours behind actual flood peaks.
Peak Smoothing Paradox: Extreme events represent a statistical minority (<13% of historical records). Symmetric loss functions optimize for the dominant baseflow regime, severely underestimating maximum flood amplitudes.
Asymmetric Gradient Loss (AGL) directly reshapes the optimization error surface during backpropagation by coupling:
Hydrograph Kinematics (Momentum): Rate-of-change sensitivity to force rapid activation during the rising limb.
Operational Risk Asymmetry: Severe penalties for peak underestimations when water levels exceed civil defense alert thresholds.
Parametric Efficiency: Enabling lightweight single-layer networks (e.g., Vanilla-LSTM) to match or outperform multi-layered state-of-the-art hybrid architectures (e.g., Hydro-Informer).
Mathematical Formulation
The AGL objective function is defined as a weighted asymmetric formulation:
L_AGL = (1 / (sum(w_i) + epsilon)) * sum(w_i * [P_base + P_asym + P_grad])


Key Mathematical Components:
Temporal Importance Weight: Amplifies the gradient step when river rising velocity exceeds the noise threshold.
Magnitude Penalty: Scales the quadratic error proportionally once water level surpasses alert stage L.
Underestimation Asymmetry: Massive penalization factor enforcing conservative, life-saving overestimation bias above hazard thresholds.
Dynamic Momentum Term: Directly attacks phase lag by penalizing deviations during rapid stage acceleration.
Experimental Framework & Methodology
Phase
Model Architecture
Target Watershed
Forecast Horizon
 
Phase 1: Loss Isolation
Multi-Input BiLSTM (AGL vs. MSE)
Rio dos Sinos (RS) & Rio Itajaí-Açu (SC)
8 Hours (T+8h)
Phase 2: Parametric Efficiency
Vanilla-LSTM + AGL vs. Hydro-Informer (SOTA)
Toplá River Basin
12 Hours (T+12h)

Key Results Summary
Basin
Model
MAE (cm)
RMSE (cm)
R²
Peak Capture
Effective Lead Time
Historic Peak Error
 
Sinos River (Peak: 811.0 cm)
Baseline (MSE)
9.91
14.28
0.96
76.19%
2.00 h
140.4 cm (Underestimated)
Sinos River (Peak: 811.0 cm)
AGL (Ours)
17.38
20.82
0.92
100.00%
6.05 h
19.1 cm
Itajaí-Açu River (Peak: 939.5 cm)
Baseline (MSE)
17.10
29.60
0.92
94.87%
8.35 h
77.3 cm (Underestimated)
Itajaí-Açu River (Peak: 939.5 cm)
AGL (Ours)
28.72
36.74
0.88
100.00%
11.33 h
16.0 cm

Python Implementation: AGL Custom Loss
import tensorflow as tf

def flood_weighted_loss_momento(alpha=1.0, beta=10.0, gamma=1000.0, L=350.0, theta=2.0, W=10.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # 1. Wavefront velocity / temporal derivative
        dy = tf.concat([[0.0], y_true[1:] - y_true[:-1]], axis=0)
        
        # 2. Dynamic weights for rising limbs
        wi = 1.0 + W * tf.cast(dy > theta, tf.float32)
        
        # 3. Penalties
        mse_term = tf.square(y_true - y_pred)
        aceleracao_term = beta * tf.nn.relu(dy) * mse_term
        base_term = 1.0 + alpha * tf.nn.relu(y_true - L)
        under_term = gamma * tf.cast(y_true > L, tf.float32) * tf.square(tf.nn.relu(y_true - y_pred))

        numerador = tf.reduce_sum(wi * (base_term * mse_term + under_term + aceleracao_term))
        denominador = tf.reduce_sum(wi) + 1e-7
        return numerador / denominador

    return loss

