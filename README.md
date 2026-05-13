# Super Mario Bros — Reinforcement Learning Agent

A PPO-based AI agent that learns to **beat Super Mario Bros across all 8 worlds** through an automated curriculum. Built from scratch using Stable Baselines3, with custom reward shaping, dynamic entropy tuning, and GPU-accelerated parallel training.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)
![SB3](https://img.shields.io/badge/Stable_Baselines3-2.8-green)

## How It Works

The agent observes 4 stacked grayscale 84×84 frames and outputs one of 7 discrete actions (via `SIMPLE_MOVEMENT`). A CNN policy network is trained with [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) across 24 parallel NES emulators.

Rather than training on all 32 levels at once (which leads to catastrophic forgetting), the agent progresses through an **automated curriculum** — starting with a single level and gradually expanding to the full game.

## Automated Curriculum

Training is split into 8 phases. The agent must achieve a target **flag rate** (% of levels completed) over a rolling window of 200 episodes before advancing. This prevents the agent from being overwhelmed by too many diverse challenges at once.

| Phase | Levels | Flag Target | Purpose |
|-------|--------|:-----------:|---------|
| **0** | 1-1 | 80% | Anchor basic flag-seeking behaviour |
| **1** | World 1 (1-1 to 1-4) | 55% | Introduce pits, pipes, and castles |
| **2** | 1-3, 1-4 | 50% | Remedial practice — precision platforming |
| **3** | 2-2 | 50% | Remedial practice — learn to swim |
| **4** | Worlds 1–2 | 50% | Integrate land + water mechanics |
| **5** | Worlds 1–3 | 45% | Expand to 12 levels |
| **6** | Worlds 1–4 | 45% | Expand to 16 levels |
| **7** | All 8 Worlds | 40% | Endgame — 31 levels (6-1 held out for testing) |

Phase advancement is fully automatic: when the flag rate threshold is met over the rolling window (with a minimum of 5M steps per phase as a guard), the training loop tears down the environments, rebuilds them with the new level set, and continues.

## Reward Design

The reward signal is shaped to teach level completion, not just forward movement:

| Signal | Value | Purpose |
|--------|------:|---------|
| Forward progress | +0.1 / pixel | Continuous gradient to move right |
| Score delta | +1.0 / 40 pts | Rewards enemy kills (+2.5 per Goomba), coins, and powerups |
| Flag bonus | +50.0 | Terminal reward for completing the level |
| Death penalty | −5.0 | Mild punishment — avoids the agent learning to hide at the start |

The death penalty is intentionally low (−5 vs the +50 flag bonus). Earlier experiments with −50 caused the agent to learn risk-averse behaviour — standing still at the start to avoid losing points rather than attempting the level.

## Dynamic Entropy

RL agents notoriously struggle with Mario's water levels because the physics change completely. The curriculum addresses this with **per-phase entropy coefficients**:

- **Land-focused phases**: Low entropy (0.03) for precise, decisive movement
- **Swimming phases (Phase 3)**: Low entropy (0.03) tuned for consistent swimming rhythms
- **Exploration phases (Phase 1)**: Higher entropy (0.05) when encountering new level types

This prevents the agent from collapsing to a single "run right and jump" policy and forces it to discover swimming mechanics.

## Performance & Hardware

Optimised for high-end consumer hardware:

| Component | Config |
|-----------|--------|
| **GPU** | RTX 5080 — all PPO network updates and inference on CUDA 12.8 |
| **CPU** | Ryzen 7 9800X3D — 24 parallel NES emulators via `SubprocVecEnv` |
| **Throughput** | ~1,400 FPS training speed |
| **Observation** | 4-frame stack of 84×84 grayscale, `channels_last` |
| **Architecture** | `CnnPolicy` (SB3 default CNN) |

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|------:|-------|
| Learning rate | 1.5e-4 | Reduced from 2.5e-4 after clip fraction spikes |
| Clip range | 0.15 | Tightened from 0.2 to prevent policy collapse |
| Batch size | 512 | GPU-optimised |
| n_steps | 2048 | Rollout length per env |
| n_epochs | 3 | PPO update epochs per rollout |
| Frame skip | 4 | Agent acts every 4th frame |
| γ (gamma) | 0.99 | Discount factor |
| GAE λ | 0.95 | Generalised Advantage Estimation |

## Project Structure

| File | Purpose |
|------|---------|
| `train.py` | Main training script — curriculum, reward shaping, environment wrappers, and PPO config |
| `watch.py` | Live viewer — renders the agent playing while training runs separately. Auto-reloads new checkpoints |
| `play.py` | Standalone evaluation — loads the latest model and plays through levels |
| `validate_reward.py` | Diagnostic tool for testing reward signal logic |
| `test_agent.py` | Agent evaluation script |

## Getting Started

### Prerequisites

- Python 3.10
- CUDA-capable GPU (tested on RTX 5080 with CUDA 12.8)

### Installation

```bash
# Clone the repo
git clone https://github.com/JoshHodes/MarioPPO.git
cd MarioPPO

# Create virtual environment
python -m venv venv310
.\venv310\Scripts\Activate.ps1    # Windows
# source venv310/bin/activate     # Linux/Mac

# Install PyTorch for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### Training

```bash
python train.py              # Resume from latest checkpoint (auto-detects GPU)
python train.py --fresh      # Start from scratch, reset curriculum to Phase 0
python train.py --low-cpu    # Reduced parallelism for constrained systems
```

### Watching the Agent

```bash
# While training is running (loads latest checkpoint automatically):
python watch.py                  # Random stage from current curriculum phase
python watch.py --stage 1-2      # Watch a specific stage
python watch.py --deterministic  # Greedy policy (no exploration noise)

# Standalone evaluation:
python play.py
```

## Technical Notes

### Pyglet HWND Overflow (Windows)
On 64-bit Windows, `play.py` and `watch.py` apply a monkey-patch to fix a pyglet 1.5.x bug where 64-bit window handles are truncated, causing `OverflowError` on render.

### NumPy Version Pin
NumPy is pinned to 1.26.4. The NES emulator wrappers (`nes-py`) use integer types that overflow under NumPy 2.x.

### Checkpoint Resumption
The curriculum phase and step counter are persisted to `models_v2/curriculum_phase.txt`. On restart, training automatically resumes from the latest `.zip` checkpoint and the correct curriculum phase — no manual intervention needed.

## License

This project is for educational and research purposes.
