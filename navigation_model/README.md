# 🤖 Navigation Model

Classifies a robot's **movement decision** from 24 ultrasonic sensor readings using four machine learning models.

## Dataset

| Property | Value |
|---|---|
| Source | [UCI Wall-Following Robot Navigation](https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data) |
| File used | `sensor_readings_24.data` |
| Instances | 5,456 |
| Features | 24 ultrasonic sensors (US1–US24, arranged 360° around the robot) |
| Task | Multi-class classification |

### Target Classes

| Class | Share |
|---|---|
| `Move-Forward` | 40.4% |
| `Sharp-Right-Turn` | 38.4% |
| `Slight-Right-Turn` | 15.1% |
| `Slight-Left-Turn` | 6.0% |

## Notebook — `nm_notebook.ipynb`

| Section | Content |
|---|---|
| **1 · Load** | Read CSV, inspect shape & missing values |
| **2 · EDA** | Class bar + pie chart · Per-class sensor averages · Sensor correlation heatmap |
| **3 · Preprocess** | `LabelEncoder` → stratified 80/20 split → `StandardScaler` |
| **4 · Train** | Decision Tree · Random Forest · SVM · KNN |
| **5 · Evaluate** | 2×2 confusion matrix grid |
| **6 · Compare** | Grouped bar: test accuracy vs 5-fold CV accuracy |
| **7 · Feature Importance** | Random Forest sensor importance chart |
| **8 · Best Model** | Full `classification_report` for the winner |

## Models

| Model | Notes |
|---|---|
| Decision Tree | `max_depth=12` to prevent overfitting |
| Random Forest | 100 trees, parallel fit |
| SVM | RBF kernel, `C=10` |
| KNN | k=5, Euclidean distance |

> **Scaling:** `StandardScaler` is applied before training — especially important for SVM and KNN which are distance-sensitive.

## Usage

```bash
# from repo root
jupyter notebook navigation_model/nm_notebook.ipynb
```

Requires: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
