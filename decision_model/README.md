# 🚕 Decision Model — Reinforcement Learning (Taxi-v3)

Autonomous decision-making via Reinforcement Learning. An agent (robot taxi) learns to pick up a passenger and drop them at the correct destination through trial and error.

## Environment

| Property | Value |
|---|---|
| Environment | `Taxi-v3` (Gymnasium) |
| Grid | 5×5 |
| State space | 500 discrete states (taxi pos × passenger loc × destination) |
| Action space | 6 — South, North, East, West, Pick-up, Drop-off |
| Reward | +20 correct drop-off · −10 illegal action · −1 per step |

## Agents Implemented

| Agent | Type | Key Idea |
|---|---|---|
| **Random Policy** | Baseline | Uniform random action — no learning |
| **Q-Learning** | Off-policy TD | Updates using `max Q(s', a')` (greedy next state) |
| **SARSA** | On-policy TD | Updates using `Q(s', a')` for the *actual* next action taken |
| **DQN** | Neural TD | 2-layer NumPy MLP + experience replay + target network |

## Notebook — `rl_notebook.ipynb`

| Section | Content |
|---|---|
| **1 · Env Overview** | Action map, reward structure, sample state |
| **2 · Random Policy** | Baseline — uniform random action for 2000 episodes |
| **3 · Q-Learning** | Off-policy TD with ε-greedy exploration, Q-table update |
| **4 · SARSA** | On-policy TD — uses actual next action for updates |
| **5 · DQN (NumPy)** | 2-layer net, experience replay (5k buffer), target network sync every 50 eps |
| **6 · Learning Curves** | Raw + 50-ep rolling mean reward; steps-per-episode efficiency |
| **7 · Model Comparison** | Last-200 mean reward, success rate, avg steps table + bar chart |
| **8 · Q-Table Heatmap** | Max Q-value and best action visualised across all 500 states |
| **9 · Greedy Evaluation** | 100-episode evaluation with no exploration — final success rate |

## Training Configuration

```python
# Tabular agents (Q-Learning, SARSA, Random)
N_EPISODES    = 1500
MAX_STEPS     = 200
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995

# DQN (fewer episodes, faster convergence loop)
DQN_EPISODES  = 500
```

## DQN Architecture (PyTorch)

```
Input  : one-hot encoding of state (500-dim)
Hidden : Linear(500→64) → ReLU → Linear(64→64) → ReLU
Output : Linear(64→6) → Q-values for each action

Optimizer              : Adam (lr=5e-3)
Loss                   : MSE (Bellman target)
Experience replay      : 3,000 transitions, batch=32
Target network sync    : every 25 episodes
Train frequency        : every 4 env steps (speed optimisation)
ε-greedy decay         : 1.0 → 0.05
```

> Requires PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Usage

```bash
# from repo root
jupyter notebook decision_model/rl_notebook.ipynb
```

Requires: `numpy`, `matplotlib`, `seaborn`, `pandas`, `gymnasium[toy-text]`

```bash
pip install "gymnasium[toy-text]"
```
