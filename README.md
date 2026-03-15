# Robot Intelligence ML

Machine learning for robot intelligence, starting with **perception**: human activity recognition (HAR) from smartphone inertial sensor data. This repository contains the perception dataset (UCI HAR–style), a Jupyter notebook pipeline for training and comparing classifiers, and instructions to use the data from **Hugging Face**.

---

## Overview

- **Perception model**: Classifies human activities (walking, sitting, standing, etc.) from 561 time/frequency features derived from accelerometer and gyroscope signals (Samsung Galaxy S II, 50 Hz).
- **Dataset**: 30 subjects, 6 activity classes, 7,352 training and 2,947 test samples. Features are normalized to `[-1, 1]`.
- **Models**: SVM (RBF), Random Forest, KNN, Logistic Regression, and a small Neural Network; preprocessing with `StandardScaler`, optional PCA; evaluation includes confusion matrices, feature importance, and cross-validation.

---

## Dataset on Hugging Face

The perception (HAR) dataset is available on **Hugging Face**. You can load it with the `datasets` library without cloning the full repo.

### Load from Hugging Face

The dataset is hosted at **[kushsaraf/robot-intelligence-dataset](https://huggingface.co/datasets/kushsaraf/robot-intelligence-dataset)**.

```python
from datasets import load_dataset

dataset = load_dataset("kushsaraf/robot-intelligence-dataset")
```

After loading, align the splits and column names with what the notebook expects (train/test feature matrices, activity labels, optional subject IDs). The notebook currently reads from the local `robot-intelligence-dataset/` folder; you can adapt the “Load the Dataset” cell to build DataFrames from the Hugging Face dataset instead.

---

## Repository structure

```
robot-intelligence-ml/
├── README.md                    # This file
├── perception_model/
│   ├── README.md                # Perception (HAR) model details
│   └── pm_notebook.ipynb        # Full pipeline: load → preprocess → train → evaluate
└── robot-intelligence-dataset/
    └── perception_dataset/      # UCI HAR–style data (on Hugging Face)
        ├── activity_labels.txt
        ├── features.txt, features_info.txt
        ├── train/               # X_train.txt, y_train.txt, subject_train.txt, Inertial Signals/
        └── test/                # X_test.txt, y_test.txt, subject_test.txt, Inertial Signals/
```

---

## Perception model (HAR)

| Item | Description |
|------|-------------|
| **Notebook** | `perception_model/pm_notebook.ipynb` |
| **Sections** | Load data → inspect → preprocess (StandardScaler, LabelEncoder) → train 5 classifiers → PCA → confusion matrix → feature importance → 5-fold CV |
| **Best reported test accuracy** | ~95.5% (Logistic Regression) |

See **`perception_model/README.md`** for dataset stats, model table, and step-by-step notebook outline.

---

## Setup and run

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
   ```

   If you use the dataset from Hugging Face:

   ```bash
   pip install datasets
   ```

3. **Run the notebook**

   Open `perception_model/pm_notebook.ipynb` in Jupyter or VS Code, select the `.venv` kernel, and run all cells. For local data, run from the repo root so that the path `../robot-intelligence-dataset/perception_dataset` resolves correctly.

---

## Dataset credit

The perception dataset is based on the **UCI Human Activity Recognition Using Smartphones Dataset**. If you use it in publications, please cite:

*Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.*

---

## License

Dataset terms follow the UCI HAR attribution above. Code in this repository is provided as-is for use with the uploaded Hugging Face dataset and local experimentation.
