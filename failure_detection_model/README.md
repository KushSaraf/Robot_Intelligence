# 🤖 Failure Detection Model

Classifies robot **execution failures** from 6-axis force/torque time-series data collected immediately after failure detection.

## Dataset

| Property | Value |
|---|---|
| Source | [UCI Robot Execution Failures](https://archive.ics.uci.edu/ml/datasets/Robot+Execution+Failures) |
| Files | `lp1.data` – `lp5.data` (5 learning problems) |
| Sensor | 6-axis F/T: Fx, Fy, Fz, Tx, Ty, Tz |
| Window | 15 time-steps per instance |
| Total instances | ~463 across all tasks |

### Learning Problems

| File | Task |
|---|---|
| `lp1` | Failures in approach to grasp position |
| `lp2` | Failures in transfer of a part |
| `lp3` | Position of part after a transfer failure |
| `lp4` | Failures in approach to ungrasp position |
| `lp5` | Failures in motion with part |

### Classes (across all tasks)
`normal` · `collision` · `obstruction` · `fr_collision` · `back_col` · `toe_up` · `slight_collision` · `bottom_collision` · `bottom_obstruction` · `collision_in_part` · `collision_in_tool`

## Feature Engineering

Each 15×6 window is flattened and augmented with per-axis statistics:

| Feature group | Count | Description |
|---|---|---|
| Raw time-steps | 90 | `t0_Fx … t14_Tz` |
| Stat features | 24 | mean, std, min, max per axis |
| **Total** | **114** | |

## Notebook — `fd_notebook.ipynb`

| Section | Content |
|---|---|
| **1 · Load & Parse** | Custom block-format parser for lp*.data files; combines all 5 tasks |
| **2 · EDA** | Class bar/breakdown · per-class mean F/T time-series · PCA scatter |
| **3 · Preprocess** | `LabelEncoder` → stratified 80/20 split → `StandardScaler` |
| **4 · Train** | Logistic Regression · Decision Tree · Random Forest · SVM · Gradient Boosting |
| **5 · Evaluate** | 2×3 confusion matrix grid for all 5 models |
| **6 · Compare** | Grouped bar: test accuracy vs 5-fold CV (with error bars) |
| **7 · Feature Importance** | Top-20 RF features; stat vs raw importance ratio |
| **8 · Best Model** | Full `classification_report` for the winner |

## Models

| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=15` |
| Random Forest | 150 trees, parallel fit |
| SVM | RBF kernel, `C=10`, `gamma='scale'` |
| Gradient Boosting | 100 estimators |

> **Scaling:** `StandardScaler` applied before training — required for Logistic Regression and SVM.

## Usage

```bash
# from repo root
jupyter notebook failure_detection_model/fd_notebook.ipynb
```

Requires: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
