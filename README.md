# Anomaly Detection and Isolation of Compromised Sensors and Actuators in Water Distribution Networks

> A Deep Learning Approach Using GRU Models and Signature Vectors on the BATADAL Dataset  
---

## Overview

Water distribution networks (WDNs) are critical infrastructure vulnerable to both cyber intrusions and physical faults. This project proposes a residual-based anomaly detection and fault isolation framework built on **Gated Recurrent Unit (GRU)** neural networks, evaluated on the [BATADAL benchmark dataset](https://www.batadal.net).

Each GRU model learns the relational behaviour of a single network component. Deviations between predicted and actual sensor readings form **residuals** that drive anomaly detection. A novel **virtual attack injection** mechanism injects synthetic sinusoidal perturbations to generate reference signatures, which are then compared against real attack patterns using **cosine similarity** to isolate the most likely compromised components.

---

## Key Features

- Per-component GRU models (NARX-style) for all 43 components (31 sensors + 12 actuators)
- Residual-based anomaly detection with data-driven, component-specific thresholds
- Virtual attack injection via sinusoidal perturbations for stress testing and signature generation
- Binary signature vectors for compact "fingerprinting" of anomalous system states
- Cosine similarity matching between real and virtual attack signatures for fault isolation
- ROC/AUC evaluation with an overall AUC of **0.819**
- Cosine similarity performance score of **0.89** (threshold-based) and **0.75** (average-residual-based)

---

## Dataset

This project uses the **BATADAL (Battle of the Attack Detection Algorithms)** benchmark datasets, simulating cyber-physical attacks on the **C-Town** water distribution network.

| Dataset | Period | Content | Purpose |
|---|---|---|---|
| `BATADAL_dataset03.csv` | 1 year | Normal operations only | GRU model training |
| `BATADAL_dataset04.csv` | ~6 months | Normal + labelled attacks | Threshold calibration & ROC evaluation |
| `BATADAL_test_dataset.csv` | 3 months | Attacks (unlabelled) | Blind anomaly detection evaluation |

The C-Town network has:
- 7 storage tanks (T1–T7)
- 11 pumping stations (PU1–PU11) and 2 valves (V1, V2)
- 31 sensors: tank levels (`L_Tx`), pressures (`P_Jxxx`), and pipe flows (`F_PUx`)
- 12 actuators: pump/valve states (`S_PUx`, `S_Vx`)

Download datasets from: [https://www.batadal.net](https://www.batadal.net)

---

## Methodology

### 1. Data Preprocessing
- Replace sentinel values (`-999`) with NaN and drop incomplete rows
- Min-Max scaling to `[0, 1]` using a scaler fit on clean Dataset03
- Sliding window sequences of length `W = 20` steps

### 2. GRU Modelling
Each component is modelled individually with a NARX-style GRU:

```
ŷ(t) = GRU( y(t-1), ..., y(t-N), u(t), u(t-1), ..., u(t-N) )
```

- Architecture: GRU (64 hidden units) → Dropout (0.2) → Dense (1)
- Loss: MSE (or Huber for noisier components)
- Optimiser: Adam (lr = 0.001), batch size 64, up to 50 epochs with early stopping

### 3. Residual Computation & Thresholding
```
r(t) = y(t) - ŷ(t)
```
Anomaly is declared when `|r(t)| > θ`, where `θ` is the 95th–99th percentile of residuals on clean training data.

### 4. Virtual Attack Injection
Sinusoidal perturbations are injected into clean data one component at a time:
```
y'(t) = y(t) + A · sin(ω · t)
```
Amplitude `A ∈ [0.1, 1.5]`, frequency `ω ∈ [0.1, 0.3]` rad/step.

### 5. Signature Vector Construction
Binary signature vectors `S ∈ {0, 1}^43` encode which components exceeded their residual threshold during an event window. These are built for both virtual and real attack periods.

### 6. Cosine Similarity Matching
```
CosSim(S_test, S_virt) = (S_test · S_virt) / (||S_test|| · ||S_virt||)
```
High similarity indicates the virtual attack targets the same component as the real anomaly, enabling **fault isolation**.

---

## Results Summary

| Metric | Value |
|---|---|
| Overall ROC AUC | 0.819 |
| Cosine similarity (threshold-based signature) | 0.89 |
| Cosine similarity (average-residual signature) | 0.75 |
| Tank sensor MSE (training) | 0.002–0.01 |

- **Tanks (L_T1–L_T7):** Lowest prediction error due to smooth dynamics
- **Pressure sensors (P_Jxxx):** Medium error from hydraulic state fluctuations
- **Pump flow sensors (F_PUx):** Higher error; improved with bidirectional GRUs + Huber loss

---

## Project Structure

```
.
├── BATADAL_dataset03.csv          # Training data (normal operations)
├── BATADAL_dataset04.csv          # Training data (with labelled attacks)
├── BATADAL_test_dataset.csv       # Test data (unlabelled)
│
├── MODEL.ipynb                    # Main notebook (all pipeline stages)
│
└── results/
    ├── *_gru_model.keras          # Trained GRU models (one per component)
    ├── minmax_scaler_dataset03.pkl
    ├── clean_thresholds.csv       # Per-sensor thresholds (sensors)
    ├── clean_thresholds_all.csv   # Per-component thresholds (all 43)
    │
    ├── virtual_attacks/           # Sinusoidally attacked inputs (sensor targets)
    ├── virtual_attacks_all/       # Sinusoidally attacked inputs (all components)
    ├── virtual_residuals/         # Residual time series under virtual attacks
    ├── virtual_residuals_all/     # Full residuals + merged averages
    │
    ├── virtual_sigvec_timeseries/ # Per-target binary signature time series
    ├── virtual_sigvec_timeseries_combined.csv
    ├── virtual_sigvec_timeseries_combined_clean.csv
    │
    ├── test_residual_plots/       # Residual plots on test data
    ├── test_residual_csv/         # Residual CSVs on test data
    ├── test_sigvec_timeseries.csv # Binary signature time series on test data
    ├── test_residuals/            # Detailed test residuals per component
    │
    ├── cosine_global_virtual_vs_test.csv
    ├── cosine_by_windows_virtual_vs_test.csv
    ├── signature_similarity_per_component.csv
    └── dataset04_summary_auc.csv
```

---

## Pipeline Walkthrough

The entire pipeline runs in sequence within `MODEL.ipynb`. The logical stages are:

| Stage | Description |
|---|---|
| **Phase 1** | Train GRU models on sensor columns from Dataset03; save models, residuals, and scaler |
| **Phase 1b** | Train GRU models for actuator columns; recompute unified thresholds for all 43 components |
| **Phase 2** | Evaluate on Dataset04 (labelled); compute AUC per sensor |
| **Phase 3a** | Generate residual plots and CSVs on the test dataset |
| **Phase 3b** | Inject virtual sinusoidal attacks on each of the 43 components; build signature vector time series |
| **Phase 3c** | Combine per-target signature time series into a single matrix |
| **Phase 3d** | Build test signature vector time series |
| **Phase 4** | Compute global and windowed cosine similarity (virtual vs. test) |
| **Phase 5** | Compute average residuals (virtual and test); compare magnitudes |
| **Phase 6** | Merge and average test residuals; produce final comparison tables |

---

## Setup & Requirements

### Environment

```bash
python >= 3.9
tensorflow >= 2.12
scikit-learn
pandas
numpy
matplotlib
joblib
```

### Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

### Run

1. Place the three BATADAL CSV files in the project root directory.
2. Open and run `MODEL.ipynb` cell by cell from top to bottom.
3. All outputs are written to the `results/` directory automatically.

> **Note:** Training all 43 GRU models is compute-intensive. GPU acceleration (e.g., via CUDA) is recommended. Running on CPU is possible but will be slow for the virtual attack injection stage.

---

## Reproducing Key Results

After running the full notebook:

- **AUC per sensor:** `results/dataset04_summary_auc.csv`
- **Cosine similarity per component:** `results/cosine_global_virtual_vs_test.csv`
- **Windowed cosine similarity (per attack interval):** `results/cosine_by_windows_virtual_vs_test.csv`
- **Test vs. virtual average residual comparison:** `results/test_residuals/test_vs_virtual_avg.csv`

---


## References

1. Taormina et al. (2018). Battle of the Attack Detection Algorithms (BATADAL). *Journal of Water Resources Planning and Management.* https://doi.org/10.1061/(ASCE)WR.1943-5452.0000969
2. Cho et al. (2014). Learning phrase representations using RNN encoder–decoder. *EMNLP.*
3. Kingma & Ba (2015). Adam: A method for stochastic optimization. *ICLR.*
4. Rahutomo et al. (2012). Semantic Cosine Similarity. *ICAST.*
5. Scikit-learn: https://scikit-learn.org

---

## License

This project was developed for academic purposes as part of SUTD course 42.705. Dataset usage is subject to the BATADAL competition terms at [https://www.batadal.net](https://www.batadal.net).
