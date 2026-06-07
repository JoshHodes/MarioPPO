# Super Mario Bros — Reinforcement Learning Agent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)
![SB3](https://img.shields.io/badge/Stable_Baselines3-2.8-green)

A deep-RL agent that learns to play Super Mario Bros **from raw pixels** — and a documented investigation into *where* model-free RL breaks on a hard, long-horizon game, and how to push past it. Built with PPO, an automated curriculum, and an imitation-learning pipeline; every result is honestly evaluated.

![Agent playing across multiple levels](mario_grid.gif)

## Highlights

- **Clears most early-game levels at 58–92%** flag-completion, learned from raw pixels (PPO + an automated curriculum across 24 parallel NES emulators).
- **Cracked a level the standard tools couldn't.** Level 1-3 (precision pit-jumps) is a textbook *hard-exploration* wall — PPO-from-scratch **and** Random Network Distillation curiosity both flat-line at **0%**. I diagnosed the failure mode, then broke through it with **imitation learning** (behavioral cloning → PPO fine-tune): **0% → 14%**.
- **Caught a bug in my own evaluation.** A "38% success" result turned out to be a selection-biased fluke that clean-evaluated to **0%**; I built a proper held-out evaluator and re-grounded every number in the repo. (The honest figure is the 14% above.)
- **Profiled and explained the bottleneck.** Proved the workload is single-thread-bound — a **256-core cloud box ran *slower*** than a desktop Ryzen 9800X3D — and tuned throughput to a ~1,400 FPS ceiling instead of buying compute that wouldn't help.
- **Diagnosed and fixed catastrophic forgetting.** A single-level "remedial" curriculum phase erased competence across *every* level in ~21M steps; the fix was a monotonically-growing level pool with weighted oversampling.

> **On the obvious question — "but does it beat Mario?"** No single model-free RL agent reliably beats Super Mario Bros *from pixels*; it's an open research problem. The famous end-to-end clears use **tree search over the emulator** (a planner with a perfect model) — **not** a learned reactive policy. Measured against that reality, this project clears most early levels and pushes a known-hard one *past* where PPO and curiosity plateau. What it deliberately doesn't do is overclaim — it's a case study in the engineering, the limits, and the honest evaluation of the method. See **[Findings & Lessons](#findings--lessons)**.

## Results

Per-level flag-completion rate of the best checkpoint (stochastic policy, ~12 episodes/level):

| Level | Flag rate | Notes |
|-------|:---------:|-------|
| 1-1 | ~67% | overworld |
| 1-2 | ~75% | underground |
| 1-3 | **0% → 14%** | precision jumps over pits — a hard-exploration wall PPO/RND never moved; **cracked with imitation learning** ([see below](#breaking-the-wall-imitation-learning)) |
| 1-4 | ~92% | castle |
| 2-3 | ~67% | overworld |
| 2-4 | ~58% | castle |
| 2-1 | **0%** | gets ~halfway then stalls |
| 2-2 | **0%** | underwater — different physics |

**Honest summary:** the agent clears overworld/castle levels with 58–92% reliability. Underwater levels (2-1, 2-2) remain at 0%. Precision-platforming level 1-3 was *also* stuck at 0% under PPO and curiosity — until imitation learning lifted it to a real (if not yet reliable) 14%, documented [below](#breaking-the-wall-imitation-learning). A full 1-1 → 8-4 single-run clear is **not** achieved — and, per the note up top, isn't a bar any model-free-RL-from-pixels agent currently meets.

## How It Works

The agent observes 4 stacked grayscale 84×84 frames and outputs one of 7 discrete actions (`SIMPLE_MOVEMENT`). A CNN policy (`CnnPolicy`) is trained with [PPO](https://arxiv.org/abs/1707.06347) across 24 parallel NES emulators (`SubprocVecEnv`). Levels are sampled from a curriculum that grows the level pool over time.

## Automated Curriculum

Training progresses through phases, each a **strict superset** of the previous one — the pool only ever grows. Weak levels are pulled up by *weighted oversampling* (`compute_weighted_stages`), never by isolating them.

> **This "never narrow the pool" rule was learned the hard way.** An earlier version included single-level "remedial" phases (e.g. train on `2-2` alone). Training on one level for ~21M steps caused **catastrophic forgetting** — the model collapsed to 0% on every level except the one it was grinding. The fix was to remove all narrowing phases. See [Findings & Lessons](#findings--lessons).

| Phase | Pool | Avg target | Per-level floor |
|------:|------|:----------:|:---------------:|
| 0 | 1-1 | 95% | — |
| 1 | World 1 | 85% | 65% |
| 2 | Worlds 1–2 | 80% | 60% |
| 3 | Worlds 1–3 | 75% | 50% |
| 4 | Worlds 1–4 | 70% | 45% |
| 5–8 | … expanding to Worlds 1–6 | 65→50% | 40→25% |
| 9 | All 8 worlds | 48% | 22% |
| 10 | Campaign (full game, 3 lives) | — | — |

A phase advances only when the rolling average **and** the weakest level both clear their thresholds (with a 5M-step minimum per phase as a guard).

## Reward Design

| Signal | Value | Purpose |
|--------|------:|---------|
| Forward progress | +0.1 / pixel | dense signal to move right |
| Score delta | +1.0 / 40 pts | enemy kills, coins, power-ups |
| Flag bonus | **+200** | terminal reward for completing a level |
| Death penalty | −5 | mild — too large and the agent hides at the start |

> The flag bonus was **raised from 50 to 200** after eval showed the agent stalling at ~⅔ of levels: forward-progress reward (~+320 over a full level) dwarfed a +50 flag bonus, so *finishing* was barely incentivized over *rushing partway and dying*. The change recovered several levels from 0% (e.g. 2-3, 2-4), though it did not crack the hard-exploration walls.

## Breaking the wall: imitation learning

Level **1-3** is the canonical hard-exploration wall: it demands a precise sequence of jumps over pits that a stochastic policy almost never stumbles onto by chance. **Two model-free approaches failed to move it off 0%** — PPO from scratch (stalls at x≈720) and PPO + RND curiosity (x≈765, never reaches the flag in 4M steps). The problem isn't *learning* the level; it's ever *seeing* a success to learn from.

Imitation learning sidesteps that: give the agent examples of the level being cleared, so it doesn't have to discover the solution from nothing.

**Pipeline** (`record_demos.py` → `behavioral_clone.py` → `finetune_bc.py` → `eval_ft.py`):

1. **Record human demos.** ~20 human clears of 1-3, captured at the agent's exact 15 Hz observation/action cadence (slow-motion playback keeps the data in-distribution). Same 84×84 grayscale, 4-stack, frameskip-4 pipeline as the PPO agent.
2. **Behavioral cloning.** Supervised-train a CNN (same NatureCNN trunk as the PPO policy) to imitate the demos. Result: pure BC **sails past the curiosity wall** — mean max-x **972 vs RND's ~765** — but never reaches the flag (0%). It plays the first ⅔ of the level correctly, then dies in the later third on states the handful of demos never covered (classic *distribution shift*).
3. **BC → PPO fine-tune.** Transfer the BC weights into a Stable-Baselines3 `CnnPolicy` (verified the SB3 net reproduces the BC net's actions on 100% of demo frames), then fine-tune with PPO and the dense shaped reward. BC already plays most of the level, so PPO only has to refine the final stretch — *local* exploration it can actually do. With annealed exploration (hold entropy high to break through, decay it to consolidate) and held-out-eval checkpointing, this reaches a genuine **14%** clear rate on a clean 50-episode evaluation.

**Honesty about the number — and the bug that produced it.** An earlier run reported a "38% best." Clean-evaluating the saved checkpoint gave **0%**: the 38% was a selection-biased peak of a noisy rolling *training* window that had already drifted by the time the model was saved. That discovery is why `eval_ft.py` exists — an independent N-episode evaluator that reports the honest flag rate, decoupled from the training loop. Re-grounded on it, the real, stable result is **14%** (`models_v2/bc_ft_1-3_best.zip`, 50 episodes, 95% CI ±9.6%). Subsequent experiments showed the annealing schedule is essentially tapped out at this level — pushing consolidation harder *regresses* to 0% — so 14% is reported as a real breakthrough over 0%, not as a solved level.

**Why this is the most important result in the repo:** it's a learned, reactive pixels→action policy that does something PPO and curiosity provably could not — broke a hard-exploration wall — arrived at through *diagnosis* (identifying *why* RL stalled) and *technique selection* (imitation), with the headline number verified rather than cherry-picked.

## Findings & Lessons

The engineering takeaways are the real substance of this project:

1. **Catastrophic forgetting is brutal and easy to trigger.** Narrowing the training distribution to a single level erased competence across all others in ~21M steps. The literature warns of this; here it is, measured. Fix: monotonically growing pools + weighted oversampling.
2. **Reward shaping must make the *goal* dominate.** A dense progress reward that out-weighs the terminal completion reward teaches the agent to farm progress, not finish. Rebalancing the flag bonus measurably recovered stuck levels.
3. **This workload is single-thread-bound.** Rollout collection is one GIL-bound Python loop stepping all envs synchronously, so throughput depends on single-core speed, **not** core count. A 256-core cloud box ran *slower* than a desktop Ryzen 9800X3D; ~1,400 FPS at 24 envs locally was the practical ceiling.
4. **Curiosity (RND) did not crack hard-exploration walls.** A controlled experiment ([`explore_rnd.py`](explore_rnd.py)) added Random Network Distillation intrinsic reward and trained 4M steps on 1-3 alone. Result: the agent went from dying at x≈720 to x≈765 and **never reached the flag once**. Strong evidence that precise-sequence levels are beyond model-free PPO + curiosity here.
5. **Imitation beat the wall curiosity couldn't.** A few demonstrations turned an unsolvable exploration problem into a tractable supervised one: BC → PPO fine-tune took 1-3 from a hard **0% to 14%** (see [Breaking the wall](#breaking-the-wall-imitation-learning)). The lesson isn't "imitation is magic" — pure BC also failed (distribution shift) — it's that the *right diagnosis* (this is an exploration failure, not a capacity failure) points to the *right tool*.
6. **Trust your eval, not your training curve.** A "38%" headline from a rolling training-window metric clean-evaluated to **0%** — a selection-biased peak the model had already drifted past by save time. Building an independent evaluator ([`eval_ft.py`](eval_ft.py)) and re-grounding every number on it is the difference between a real result and a self-deluding one.

## Limitations (what doesn't work, and why)

- **Hard-exploration levels stay at 0%.** 1-3 (precise jumps over pits) and 2-1/2-2 (water) require specific action sequences a stochastic policy almost never stumbles onto. Curiosity didn't fix it.
- **Single-network interference.** One small CNN holding many distinct levels trades off competence — pushing hard levels up tends to drag easy ones down.
- **No memory / no planning.** A reactive feed-forward policy can't plan through World 8-4's pipe-maze, and PPO learns only by stumbling onto rewarded behavior.
- **A full single-run clear compounds all of the above.** Beating 1-1 → 8-4 on 3 shared lives needs ~75%+ reliability on *every* level simultaneously — far beyond what this method reaches.

See [How this *could* be achieved](#how-a-full-clear-could-actually-be-achieved) for approaches better suited to the goal.

## How a full clear could actually be achieved

Beating Mario is fundamentally a **search/planning** problem, and a perfect simulator (the emulator) is available — which is exactly what model-free PPO doesn't exploit. More suitable approaches:

- **Search over the emulator (A\* / MCTS / TAS-style).** With the emulator as a perfect model, tree search over save-states can plan action sequences that clear *any* level, including 8-4, deterministically. (Cf. Robin Baumgarten's A\* Mario.) This is the reliable way to *guarantee* a clear — it just isn't a learned reactive policy.
- **Go-Explore.** Purpose-built for hard exploration: remember promising states, return to them, explore onward, then robustify via imitation. It solved Montezuma's Revenge / Pitfall and has been demonstrated on Mario.
- **Imitation / offline RL from demonstrations.** A few human (or scripted) clears bypass the exploration problem; behavioral-clone then RL-fine-tune. **Demonstrated here** — it took 1-3 from 0% to 14% ([Breaking the wall](#breaking-the-wall-imitation-learning)). The remaining gap to *reliable* is a coverage/robustness problem (more targeted demos, DAgger, or a BC-anchored PPO objective), not an exploration one — which is exactly the right kind of problem to be left with.
- **Model-based RL (MuZero / DreamerV3).** Learn a world model and *plan* — far stronger on hard exploration and sample efficiency than model-free PPO.

## Getting Started

### Prerequisites
- Python 3.10, a CUDA-capable GPU.

### Install
```bash
python -m venv venv310 && source venv310/bin/activate   # Windows: .\venv310\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Train
```bash
python train.py                          # resume from latest checkpoint
python train.py --fresh                  # start from Phase 0
python train.py --resume-from path.zip   # resume from a specific checkpoint
```

### Evaluate
```bash
python evaluate.py --episodes 12                       # per-level flag rates
python evaluate.py --campaign --episodes 50            # full-game (3 lives) probe
```

### Watch / experiments
```bash
python watch.py                          # render the agent playing the latest checkpoint
python brute_force_campaign.py           # sample many stochastic full-game runs, save the best
python explore_rnd.py --level 1-3        # RND curiosity exploration experiment
```

### Imitation pipeline (the 1-3 breakthrough)
```bash
python record_demos.py --level 1-3                       # record human clears -> demos/1-3.npz
python behavioral_clone.py --level 1-3                   # behavioral cloning -> models_v2/bc_1-3.pt
python finetune_bc.py --level 1-3                        # BC -> PPO fine-tune
python eval_ft.py --model models_v2/bc_ft_1-3_best.zip --level 1-3 --episodes 50   # honest flag rate
```

## Project Structure

| File | Purpose |
|------|---------|
| `train.py` | Curriculum, reward shaping, env wrappers, PPO config |
| `evaluate.py` | Per-level and full-campaign evaluation |
| `brute_force_campaign.py` | Parallel stochastic full-game sampler; records the best/winning run |
| `explore_rnd.py` | RND intrinsic-curiosity exploration experiment |
| **Imitation pipeline** | |
| `record_demos.py` | Human-play recorder; captures demos at the agent's 15 Hz cadence |
| `bc_common.py` | Shared obs pipeline + `BCPolicy` (NatureCNN) + headless flag-rate evaluator |
| `behavioral_clone.py` | Class-weighted behavioral cloning on demos |
| `finetune_bc.py` | Transfer BC weights into SB3 PPO and fine-tune (annealed exploration, real-eval checkpointing) |
| `eval_ft.py` | Independent N-episode evaluator — the honest flag-rate number |
| `watch.py` / `play.py` | Render the agent playing |
| `plot_training.py` / `parse_logs.py` | TensorBoard plotting / metric readout |

## Technical Notes

- **NumPy pinned to 1.26.4** — the NES wrappers (`nes-py`) overflow under NumPy 2.x.
- **Checkpoint resumption** — curriculum phase and step counter persist in `models_v2/curriculum_phase.txt`.

## License

For educational and research purposes.
