# Multi-Agent-Reinforecement-Learning-Swarm-Navigation

> A multi-agent reinforcement learning project where a swarm of agents learns to reach their goals while avoiding collisions with each other and with moving obstacles — trained using **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient).

Four agents each get their own goal in a shared 2D arena. They have to learn to steer toward their goal while staying clear of teammates and unpredictable dynamic obstacles. The project includes training, a 2D viewer, and a 3D viewer with an extra safety layer.

---

## Demo

The project ships with two ways to watch trained agents in action:

- **2D viewer** (`visualize.py`) — a fast top-down view built with Pygame, with motion trails, safety zones, velocity arrows, and a live stats panel.
- **3D viewer** (`visualize_3d.py`) — a 3D scene built with PyBullet that adds a reactive safety filter and continuous online learning on top of the trained policy.

---

## What is MADDPG?

MADDPG is a reinforcement learning algorithm designed for environments with multiple agents. The key idea is **centralized training, decentralized execution**:

- Each agent has its own **actor** network that decides what to do using only that agent's local view of the world.
- During training, each agent also has a **critic** network that sees *every* agent's observations and actions. This shared view makes learning stable even though the other agents are also changing.
- Once trained, agents act independently using just their actors — no central controller needed.

---

## Features

- **MADDPG implementation** in PyTorch — separate actor and critic networks per agent, with target networks and soft updates.
- **Custom swarm environment** — a 2D world with agents, per-agent goals, and dynamic obstacles that move in circular or random patterns.
- **Shaped reward function** — dense progress reward toward the goal, a bonus for reaching it, and graduated penalties for getting too close to other agents or obstacles.
- **Experience replay buffer** for stable off-policy learning.
- **TensorBoard logging** of episode reward and goals reached.
- **2D Pygame visualizer** with interactive controls (pause, speed, trails, safety zones).
- **3D PyBullet visualizer** with a potential-field **safety filter** and **online learning** that keeps improving the policy live and logs every collision.

---

## Project structure

```
.
├── config.py             # All hyperparameters and environment settings
├── train.py              # Training loop — trains the swarm and saves checkpoints
├── visualize.py          # 2D Pygame viewer for a trained model
├── visualize_3d.py       # 3D PyBullet viewer with safety filter + online learning
├── env/
│   ├── __init__.py
│   └── swarm_env.py       # The swarm environment, agents, and dynamic obstacles
├── agents/
│   ├── __init__.py
│   ├── actor.py           # Actor network (decides actions from local observation)
│   ├── critic.py          # Critic network (estimates value from global state)
│   ├── maddpg.py          # MADDPG algorithm — ties actors/critics together
│   └── replay_buffer.py   # Experience replay buffer
└── checkpoints/
    ├── best_model.pt           # Best model from training
    └── best_model_finetuned.pt # Model after online fine-tuning (3D viewer)
```

> **Note:** the two `__init__.py` files are empty on purpose — they tell Python that `env/` and `agents/` are importable packages.

---

## Installation

Requires **Python 3.9+**.

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy tensorboard pygame pybullet
```

---

## Usage

### Train the swarm

```bash
python train.py
```

This runs the training loop for `MAX_EPISODES` episodes. Progress prints every 100 episodes, and the best-performing model is saved to `checkpoints/best_model.pt`. Training metrics are written to `runs/` for TensorBoard.

To watch the training curves:

```bash
tensorboard --logdir runs
```

### Watch the trained agents (2D)

```bash
python visualize.py
```

If a trained model exists in `checkpoints/`, it is loaded automatically; otherwise an untrained policy is shown.

**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | pause / play |
| `+` / `-` | speed up / slow down |
| `R` | reset episode |
| `T` | toggle motion trails |
| `S` | toggle safety zones |
| `V` | toggle velocity arrows |
| `Q` | quit |

### Watch the trained agents (3D)

```bash
python visualize_3d.py
```

The 3D viewer adds two extra layers on top of the trained policy:

1. **Safety filter** — a potential-field overlay that pushes each agent away from nearby agents and obstacles *before* its action is executed, sharply reducing collisions even when the learned policy is imperfect.
2. **Online learning** — after a warmup period, the MADDPG update loop runs live so the policy keeps improving. Improvements are saved to `checkpoints/best_model_finetuned.pt` (the original `best_model.pt` is left untouched). Every collision is logged to `logs/collisions.jsonl`.

---

## How the environment works

- **Agents** — each agent is a point that moves with a 2D velocity action and has its own goal.
- **Goals** — random target positions; an agent "reaches" its goal when it gets within a small radius.
- **Dynamic obstacles** — moving hazards that follow either a circular or a random walk pattern and bounce off the world bounds.
- **Observation** — each agent sees its own position and velocity, the direction to its goal, the relative positions of the other agents, and the relative positions and velocities of all obstacles.
- **Reward** — agents earn reward for making progress toward their goal and a bonus for reaching it, and lose reward for getting dangerously close to other agents or obstacles.

---

## Configuration

All settings live in `config.py`. Key values:

| Setting | Default | Meaning |
|---------|---------|---------|
| `N_AGENTS` | `4` | Number of agents in the swarm. |
| `N_DYN_OBSTACLES` | `3` | Number of moving obstacles. |
| `HIDDEN_DIM` | `256` | Size of the network hidden layers. |
| `LR_ACTOR` / `LR_CRITIC` | `1e-4` / `3e-4` | Learning rates. |
| `GAMMA` | `0.95` | Discount factor for future rewards. |
| `TAU` | `0.01` | Soft-update rate for target networks. |
| `BATCH_SIZE` | `256` | Replay batch size. |
| `BUFFER_SIZE` | `100000` | Replay buffer capacity. |
| `MAX_EPISODES` | `5000` | Number of training episodes. |
| `MAX_STEPS` | `50` | Steps per episode. |
| `EXPLORATION_NOISE` | `0.2` | Action noise during training (for exploration). |

---

## Limitations

- The environment is a simplified 2D abstraction — agents are points, not physically simulated drones.
- Training time depends on your hardware; a GPU speeds things up considerably.
- This is a learning/research project, not production-grade swarm control software.

---

## Acknowledgements

- [PyTorch](https://pytorch.org/) for the neural networks and training.
- [Pygame](https://www.pygame.org/) for the 2D visualizer.
- [PyBullet](https://pybullet.org/) for the 3D visualizer.
- The MADDPG algorithm is based on the paper *"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"* (Lowe et al., 2017).
