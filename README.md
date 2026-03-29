# Robot Intelligence ML

Machine learning for robot intelligence across four tasks — **perception** (human activity recognition), **navigation** (movement decisions from ultrasonic sensors), **failure detection** (classifying robot execution failures from force/torque signals), and **autonomous decision-making** (reinforcement learning in a simulated environment). Each task has its own dataset, notebook pipeline, and model comparison.

---

## Overview

- **Perception model**: Classifies human activities (walking, sitting, standing, etc.) from 561 time/frequency features derived from accelerometer and gyroscope signals (Samsung Galaxy S II, 50 Hz).
  - 30 subjects · 6 activity classes · 7,352 train / 2,947 test samples · features normalized to `[-1, 1]`
  - Models: SVM, Random Forest, KNN, Logistic Regression, Neural Network

- **Navigation model**: Classifies a mobile robot's movement decision from 24 ultrasonic distance sensors arranged 360° around its waist.
  - 5,456 samples · 4 classes (`Move-Forward`, `Sharp-Right-Turn`, `Slight-Right-Turn`, `Slight-Left-Turn`)
  - Models: Decision Tree, Random Forest, SVM, KNN; includes EDA, sensor correlation, feature importance, 5-fold CV

- **Failure detection model**: Classifies robot execution failures from 6-axis force/torque time-series (15 samples after failure detected).
  - ~463 instances · 5 learning problems (lp1–lp5) · up to 11 failure classes
  - Models: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting; 114 features (raw + stats)

- **RL decision model**: Agent learns to pick up and drop off a passenger in a grid-world via trial and error.
  - Environment: `Taxi-v3` (Gymnasium) · 500 states · 6 actions
  - Agents: Random baseline · Q-Learning · SARSA · DQN (NumPy, from scratch)

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
├── README.md                      # This file
├── perception_model/
│   ├── README.md                  # Perception (HAR) model details
│   └── pm_notebook.ipynb          # load → preprocess → train 5 models → evaluate
├── navigation_model/
│   ├── README.md                  # Navigation model details
│   └── nm_notebook.ipynb          # EDA → preprocess → train 4 models → compare
├── failure_detection_model/
│   ├── README.md                  # Failure detection model details
│   └── fd_notebook.ipynb          # parse → EDA → train 5 models → compare
├── decision_model/
│   ├── README.md                  # RL decision model details
│   └── rl_notebook.ipynb          # env → random → Q-Learning → SARSA → DQN → compare
├── report_website/
│   └── index.html                 # Interactive project report site (open in browser)
└── robot-intelligence-dataset/
    ├── perception_dataset/        # UCI HAR–style data (on Hugging Face)
    │   ├── activity_labels.txt
    │   ├── features.txt, features_info.txt
    │   ├── train/                 # X_train.txt, y_train.txt, subject_train.txt, Inertial Signals/
    │   └── test/                  # X_test.txt, y_test.txt, subject_test.txt, Inertial Signals/
    ├── navigation_dataset/        # UCI Wall-Following Robot Navigation data
    │   ├── Wall-following.names
    │   ├── sensor_readings_24.data  # 24 ultrasonic sensors (used by nm_notebook)
    │   ├── sensor_readings_4.data
    │   └── sensor_readings_2.data
    └── failure_dataset/           # UCI Robot Execution Failures
        ├── lp1.data               # approach to grasp
        ├── lp2.data               # transfer of part
        ├── lp3.data               # position after transfer failure
        ├── lp4.data               # approach to ungrasp
        └── lp5.data               # motion with part
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

## Navigation model (Wall-Following)

| Item | Description |
|------|-------------|
| **Notebook** | `navigation_model/nm_notebook.ipynb` |
| **Sections** | Load data → EDA (class distribution, per-class sensor averages, correlation heatmap) → preprocess (StandardScaler) → train 4 classifiers → confusion matrices → test vs CV comparison → Random Forest feature importance → best-model report |
| **Dataset** | `sensor_readings_24.data` — 5,456 samples, 24 ultrasonic sensors |
| **Classes** | `Move-Forward`, `Sharp-Right-Turn`, `Slight-Right-Turn`, `Slight-Left-Turn` |

See **`navigation_model/README.md`** for full dataset stats, model table, and notebook outline.

---

## Failure detection model (Robot Execution Failures)

| Item | Description |
|------|-------------|
| **Notebook** | `failure_detection_model/fd_notebook.ipynb` |
| **Sections** | Parse lp*.data files → EDA (class dist., time-series plots, PCA) → preprocess → train 5 classifiers → confusion matrices → test vs CV comparison → RF feature importance → best-model report |
| **Dataset** | `lp1–lp5.data` — ~463 instances, 15×6 F/T window → 114 features |
| **Classes** | `normal`, `collision`, `obstruction`, `fr_collision`, `back_col`, `toe_up`, `slight_collision`, `bottom_collision`, `bottom_obstruction`, `collision_in_part`, `collision_in_tool` |

See **`failure_detection_model/README.md`** for full dataset stats, feature engineering details, and notebook outline.

---

## RL decision model (Taxi-v3)

| Item | Description |
|------|-------------|
| **Notebook** | `decision_model/rl_notebook.ipynb` |
| **Environment** | `Taxi-v3` — 5×5 grid, 500 states, 6 actions |
| **Sections** | Env overview → Random baseline → Q-Learning → SARSA → DQN → learning curves → comparison → Q-table heatmap → greedy evaluation |
| **Agents** | Random · Q-Learning (off-policy TD) · SARSA (on-policy TD) · DQN (NumPy MLP + replay buffer + target net) |

See **`decision_model/README.md`** for agent details, DQN architecture, and training config.

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

   For the RL notebook:

   ```bash
   pip install "gymnasium[toy-text]"
   ```

   If you use the dataset from Hugging Face:

   ```bash
   pip install datasets
   ```

3. **Run the notebooks**

   Open either notebook in Jupyter or VS Code, select the `.venv` kernel, and run all cells. Always launch Jupyter from the repo root so that relative dataset paths resolve correctly.

   ```bash
   jupyter notebook perception_model/pm_notebook.ipynb
   jupyter notebook navigation_model/nm_notebook.ipynb
   jupyter notebook failure_detection_model/fd_notebook.ipynb
   jupyter notebook decision_model/rl_notebook.ipynb
   ```

---

## Dataset credit

The perception dataset is based on the **UCI Human Activity Recognition Using Smartphones Dataset**. If you use it in publications, please cite:

*Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.*

---

## License

Dataset terms follow the UCI HAR attribution above. Code in this repository is provided as-is for use with the uploaded Hugging Face dataset and local experimentation.
