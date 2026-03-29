# Perception Model — Human Activity Recognition

A machine learning pipeline that classifies human activities from smartphone sensor data using the **UCI HAR Dataset**.

---

## Dataset

**Source:** UCI Human Activity Recognition Using Smartphones Dataset  
**Path:** `../datasets/perception_dataset/`

| Property | Value |
|---|---|
| Subjects | 30 volunteers (age 19–48) |
| Sensor | Samsung Galaxy S II (accelerometer + gyroscope) |
| Sampling rate | 50 Hz |
| Features | 561 (time & frequency domain) |
| Train samples | 7,352 |
| Test samples | 2,947 |

**6 Activity Classes:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

All features are pre-normalised to `[-1, 1]`.

---

## Notebook: `pm_notebook.ipynb`

Run with the `.venv` kernel (VS Code: **Select Kernel → .venv**).

### Sections

| # | Section | Description |
|---|---|---|
| 1 | Imports | numpy, pandas, matplotlib, seaborn, scikit-learn |
| 2 | Data Loading | Load train/test splits → build `train_df`, `test_df`, `df` |
| 3 | Dataset Inspection | Feature names, value range, sample rows, activity counts |
| 4 | Dataset Overview | Shape, statistics, missing value check |
| 5 | Preprocessing | Quality check → `LabelEncoder` → `StandardScaler` (fit on train only) |
| 6 | Model Training | Train 5 classifiers, comparison table + classification report + bar chart |
| 7 | PCA Feature Reduction | 561 → ~N features (95% variance retained), accuracy with vs without PCA |
| 8 | Confusion Matrix | Seaborn heatmap for the best model |
| 9 | Feature Importance | Top-20 Random Forest feature bar chart |
| 10 | Cross-Validation | 5-fold CV mean ± std for all 5 models |

### Models & Results

| Model | Train Acc | Test Acc |
|---|---|---|
| SVM (RBF, C=10) | 0.9981 | 0.9542 |
| Random Forest (200 trees) | 1.0000 | 0.9291 |
| KNN (k=7) | 0.9785 | 0.8887 |
| **Logistic Regression** | **0.9962** | **0.9549** |
| Neural Network (256→128) | 0.9990 | 0.9450 |

**Best model:** Logistic Regression — test accuracy ≈ 95.5%

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
```

Open `pm_notebook.ipynb` in VS Code and select the `.venv` kernel.