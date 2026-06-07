import os
import warnings
# Suppress np.bool8 DeprecationWarning from gym's passive_env_checker.
# It fires on every step across all 24 subprocesses, flooding the terminal
# and slowing rollout collection through I/O contention.
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        message='.*np\\.bool8.*')
import cv2
import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import glob
import torch
import time
import argparse
from collections import deque

# Auto-detect GPU — RTX 5080 (sm_120) requires PyTorch 2.7+ / cu128.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"[GPU] {gpu_name} ({gpu_mem} GB VRAM) -- using CUDA")
    # Restore higher thread count for max performance.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
else:
    print("[WARN] No CUDA GPU found -- falling back to CPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

# Thermal throttle: seconds to sleep after every PPO update iteration.
THROTTLE_PAUSE_S = 0.5

# ─── CURRICULUM ───────────────────────────────────────────────────────────────
# Phased expansion. Advances automatically when the agent achieves
# FLAG_RATE_THRESHOLD flag-completion rate over FLAG_RATE_WINDOW episodes,
# having spent at least MIN_STEPS_PER_PHASE steps in the current phase.

CURRICULUM_STAGES = [
    # Phase 0 — 1-1 only: anchor flag-seeking behaviour from scratch fast.
    ['1-1'],
    # Phase 1 — Full World 1
    ['1-1', '1-2', '1-3', '1-4'],
    # ── REMOVED: the single-level "remedial" phases ([1-3,1-4] and [2-2]). ──────
    # Training on a narrow pool caused CATASTROPHIC FORGETTING: grinding [2-2]
    # alone collapsed the model to 0% on every other level. Every phase is now a
    # strict superset of the previous; weak levels are pulled up by weighted
    # oversampling (compute_weighted_stages), never by isolating them.
    # Phase 2 — Worlds 1-2 (Full integration)
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4'],
    # Phase 5 — Worlds 1-3
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4'],
    # Phase 6 — Worlds 1-4
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4'],
    # ── From here on: levels are ONLY EVER ADDED, never removed. ──────────────
    # Catastrophic forgetting experiments proved that narrowing the level pool
    # (even to a 2-level warm-up) destroys existing competence faster than it
    # builds new competence. Every phase below is a strict superset of phase 6.

    # Phase 7 — Introduce 5-1 into the full W1-4 pool.
    # The agent has ~51% flag rate on W1-4; adding one unfamiliar level drops
    # the ceiling slightly. Target 45% forces real 5-1 improvement.
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1'],
    # Phase 8 — Add remaining non-autoscroll W5 levels (5-2, 5-4).
    # 5-3 (autoscroll) still excluded — distinct mechanic, introduced next.
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1', '5-2', '5-4'],
    # Phase 9 — Add autoscroll (5-3). Full W1-5 pool.
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1', '5-2', '5-3', '5-4'],
    # Phase 10 — Add W6 (6-1 held out for testing). Full W1-6 pool.
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1', '5-2', '5-3', '5-4',
     '6-2', '6-3', '6-4'],
    # Phase 11 — All 8 worlds. 6-1 still held out for testing.
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1', '5-2', '5-3', '5-4',
     '6-2', '6-3', '6-4',
     '7-1', '7-2', '7-3', '7-4',
     '8-1', '8-2', '8-3', '8-4'],
    # Phase 12 — CAMPAIGN MODE: SuperMarioBros-v0 full game, 3 shared lives,
    # sequential 1-1 → 8-4. Random-stage training optimizes per-level flag
    # rate; campaign training is what teaches "don't waste lives, level N+1
    # still matters." None signals "use the campaign env" in make_venv.
    None,
]

# Per-phase flag rate thresholds.
# RAISED in v3 (campaign-aware curriculum). Old thresholds optimized for
# "average per-level flag rate." That's not the real goal: beating the full
# game in one run with 3 lives requires ~95%+ per-level reliability. Average
# in the 30-50% range guarantees the campaign phase has no foundation.
# Lower thresholds *let* the agent advance onto wobbly ground; raising them
# forces real consolidation before each expansion.
FLAG_RATE_THRESHOLDS = [
    0.95,  # Phase 0:  1-1 only        (must be near-perfect — anchor behaviour)
    0.85,  # Phase 1:  World 1         (real mastery)
    0.80,  # Phase 2:  Worlds 1-2      (land + water mechanics)
    0.75,  # Phase 5:  Worlds 1-3
    0.70,  # Phase 6:  Worlds 1-4
    0.65,  # Phase 7:  W1-4 + 5-1
    0.55,  # Phase 8:  W1-4 + 5-1/2/4
    0.55,  # Phase 9:  W1-5 full
    0.50,  # Phase 10: W1-6
    0.48,  # Phase 11: All 8 worlds    (modest gate — goal is to REACH campaign
           #           mode, where life-economy is learned; campaign mode keeps
           #           playing all levels so per-level skill still improves there)
    2.00,  # Phase 12: campaign mode   (unreachable — final phase, train indefinitely)
]
# Per-phase entropy coefficients.
ENT_COEFS = [
    0.03,  # Phase 0:  1-1 only     (low — exploit the flag signal)
    0.05,  # Phase 1:  World 1      (slight boost for harder levels)
    0.03,  # Phase 2:  Worlds 1-2   (low — precision across all mechanics)
    0.03,  # Phase 5:  Worlds 1-3   (low entropy)
    0.03,  # Phase 6:  Worlds 1-4   (low entropy)
    0.03,  # Phase 7:  W1-4 + 5-1  (same as phase 6 — W1-4 anchor keeps signal stable, no exploration boost needed)
    0.05,  # Phase 8:  W1-4 + W5   (keep elevated while consolidating W5)
    0.05,  # Phase 9:  W1-5 full   (autoscroll needs some exploration)
    0.05,  # Phase 10: W1-6        (new W6 territory)
    0.05,  # Phase 11: All worlds  (variety across W7-8)
    0.02,  # Phase 12: campaign    (low — exploit, don't gamble lives)
]
FLAG_RATE_WINDOW    = 200    # rolling window: number of completed episodes
MIN_STEPS_PER_PHASE = 5_000_000  # guard against advancing before window fills; flag rate is the real gate

# Per-phase per-level minimum flag rate. 0.0 disables the gate. Phases with
# many levels gate on weak-spot completion: a 55% average isn't allowed to
# mask one level sitting at 5%. Tuned a bit below the average target —
# strict enough to catch dead zones, lenient enough to not deadlock.
PER_LEVEL_MIN_RATES = [
    0.0,   # Phase 0:  single level — no gate needed
    0.65,  # Phase 1:  W1 — every level should be solidly playable
    0.60,  # Phase 2:  Worlds 1-2
    0.50,  # Phase 3:  Worlds 1-3
    0.45,  # Phase 4:  Worlds 1-4
    0.40,  # Phase 5:  W1-4 + 5-1
    0.30,  # Phase 8
    0.30,  # Phase 9
    0.25,  # Phase 10
    0.22,  # Phase 11  — for the brute-force-clear goal the floor only needs to be
           #             >0 (no hard-zero level, or P(full run)=0 exactly). Kept
           #             modest so 8-4 doesn't deadlock the path into campaign mode.
           #             The rebalancer (compute_weighted_stages) still hammers
           #             the weakest level 6× to pull it off zero.
    0.0,   # Phase 12: campaign mode — different metric
]

# ─── REWARD ───────────────────────────────────────────────────────────────────

class MarioReward(gym.Wrapper):
    """
    Game-beating reward signal (adapted from successful GitHub implementations):
      - Forward progress: +0.1 per pixel of x movement.
      - Score delta: +1.0 per 40 points (gives +2.5 for a Goomba kill). Teaches enemy stomping naturally.
      - Powerup state: +15 for gaining (mushroom/flower), -20 for losing. A
        powerup is a damage buffer — surviving a hit instead of dying directly
        raises completion rate. Score already rewards the pickup; the loss
        penalty is the key signal that teaches the agent to PROTECT its buffer.
      - Death penalty: -50.
      - Flag bonus: +50 (reduced from 300 to avoid destabilizing gradients).
    """
    POWER = {'small': 0, 'tall': 1, 'fireball': 2}

    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0
        self.curr_score = 0
        self.prev_power = 0

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        shaped_reward = 0.0

        # 1. Forward progress
        if 'x_pos' in info:
            x_pos = info['x_pos']
            x_delta = max(min(x_pos - self.prev_x, 15), -15)
            shaped_reward += x_delta * 0.1
            self.prev_x = x_pos

        # 2. Score delta (killing enemies, coins)
        if 'score' in info:
            score_delta = info['score'] - self.curr_score
            shaped_reward += score_delta / 40.0
            self.curr_score = info['score']

        # 3. Terminal states
        done = terminated or truncated

        # 2b. Powerup state (damage buffer). Gate on not-done so a death's
        # status->small reset isn't double-penalized with the death penalty.
        power = self.POWER.get(info.get('status', 'small'), 0)
        if not done:
            dpow = power - self.prev_power
            if dpow > 0:
                shaped_reward += 15.0 * dpow      # grabbed mushroom/flower
            elif dpow < 0:
                shaped_reward += 20.0 * dpow      # got hit, lost the buffer (dpow<0)
        self.prev_power = power

        if done:
            if info.get('flag_get', False):
                # Raised 50 -> 200. Forward progress (+0.1/px) accumulates ~+320
                # over a full level, so an old +50 flag bonus barely rewarded
                # FINISHING over rushing to ~2/3 and dying — the exact failure
                # seen in eval (0% flag despite reaching 1/3-2/3 of W2-5 levels).
                # +200 makes completing a level clearly worth more than partial
                # progress, without the gradient blow-up of the original +300.
                shaped_reward += 200.0
            else:
                # Mild death penalty (-5): too large and the agent learns to hide
                # at the start rather than risk progress.
                shaped_reward -= 5.0

        if len(result) == 5:
            return obs, shaped_reward, terminated, truncated, info
        return obs, shaped_reward, done, info

    def reset(self, **kwargs):
        self.prev_x = 0
        self.curr_score = 0
        self.prev_power = 0
        return self.env.reset(**kwargs)


class MarioCampaignReward(gym.Wrapper):
    """
    Reward shaping for CAMPAIGN mode (SuperMarioBros-v0, full game, 3 shared
    lives, sequential 1-1 → 8-4). The single-episode dynamics are completely
    different from random-stage training: dying mid-run costs you future
    stages, so the reward signal must teach risk-aware play.

      - Forward progress: +0.1 per pixel (clipped; credit suppressed across
        stage transitions and life-loss respawns so position resets don't
        register as huge negative deltas).
      - Score delta: +1.0 per 40 pts (Goombas, coins, powerups).
      - Stage clear: +50 × (1 + 0.5·(world−1)). Later worlds are worth more,
        creating a steepening reward curve that rewards survival depth.
      - Powerup state: +15 gain / -20 loss (a damage buffer; surviving a hit
        instead of losing a finite campaign life is worth a lot here).
      - Life lost: −25 (a life is finite — only 3 in the whole campaign).
      - Beat 8-4: +500 terminal bonus AND terminate (don't let the env loop
        into NES Hard Mode).
      - Game over (out of lives): −25.
    """
    POWER = {'small': 0, 'tall': 1, 'fireball': 2}

    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0
        self.prev_score = 0
        self.prev_life = 2          # NES: 2 reserves + current = 3 total lives
        self.prev_world = 1
        self.prev_stage = 1
        self.prev_power = 0
        self.stages_cleared = 0

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, _, terminated, truncated, info = result
        else:
            obs, _, done, info = result
            terminated, truncated = done, False

        shaped = 0.0

        life   = info.get('life',   self.prev_life)
        world  = info.get('world',  self.prev_world)
        stage  = info.get('stage',  self.prev_stage)
        x_pos  = info.get('x_pos',  self.prev_x)
        score  = info.get('score',  self.prev_score)
        flag   = info.get('flag_get', False)

        # Stage transition vs life loss. 0xff is the "no lives left" sentinel.
        stage_changed = (world != self.prev_world) or (stage != self.prev_stage)
        life_lost = (
            life < self.prev_life
            and self.prev_life != 0xff
            and life != 0xff
        )

        # Forward-progress credit is suppressed when position would reset
        # (stage transition or respawn). Otherwise huge negative x_delta on
        # respawn would dominate the death penalty.
        if not (stage_changed or life_lost):
            x_delta = max(min(x_pos - self.prev_x, 15), -15)
            shaped += x_delta * 0.1

        # Score is monotone within a campaign — only credit non-negative deltas.
        score_delta = score - self.prev_score
        if score_delta > 0:
            shaped += score_delta / 40.0

        # Powerup state (damage buffer). Suppress on life loss / stage change so
        # the status->small reset there isn't counted as "losing" a powerup.
        power = self.POWER.get(info.get('status', 'small'), 0)
        if not (life_lost or stage_changed):
            dpow = power - self.prev_power
            if dpow > 0:
                shaped += 15.0 * dpow
            elif dpow < 0:
                shaped += 20.0 * dpow

        if life_lost:
            shaped -= 25.0

        if flag:
            self.stages_cleared += 1
            shaped += 50.0 * (1.0 + 0.5 * (world - 1))
            if world == 8 and stage == 4:
                shaped += 500.0
                terminated = True   # full campaign cleared — stop here

        done = terminated or truncated
        if done and life == 0xff:
            shaped -= 25.0

        info = dict(info)
        info['stages_cleared'] = self.stages_cleared
        info['campaign_cleared'] = (world == 8 and stage == 4 and flag)

        self.prev_life  = life
        self.prev_world = world
        self.prev_stage = stage
        self.prev_x     = x_pos
        self.prev_score = score
        self.prev_power = power

        if len(result) == 5:
            return obs, shaped, terminated, truncated, info
        return obs, shaped, done, info

    def reset(self, **kwargs):
        self.prev_x = 0
        self.prev_score = 0
        self.prev_life = 2
        self.prev_world = 1
        self.prev_stage = 1
        self.prev_power = 0
        self.stages_cleared = 0
        return self.env.reset(**kwargs)


# ─── WRAPPERS ─────────────────────────────────────────────────────────────────

class SkipFrame(gym.Wrapper):
    """Apply action for `skip` frames, sum rewards."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated = done
                truncated = False
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class MarioWrapper(gym.Env):
    """Grayscale + resize to 84×84, gymnasium-compatible API."""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return self.process_obs(obs), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        return self.process_obs(obs), reward, terminated, truncated, info

    def process_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, axis=-1)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# ─── CALLBACKS ────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Tracks flag-completion rate over a rolling window of episodes AND
    per-level flag rates for diagnosing weak spots. When rate >=
    flag_rate_threshold (after MIN_STEPS_PER_PHASE steps), and (optionally)
    every individual level meets `per_level_min_rate`, requests an advance.

    Per-level minimum is what stops the "averages hide weak levels" failure
    mode: a 60% average can mask 5-2 sitting at 5% and 1-1 at 95%.

    Campaign mode (Phase 12) logs different metrics: per-episode stages
    cleared and full-game completion rate. Advance gate is set to 2.0 so
    it never triggers.
    """
    def __init__(self, flag_rate_threshold, window_size, min_steps_per_phase,
                 per_level_min_rate=0.0, per_level_min_samples=20,
                 campaign_mode=False, rebalance_interval=2000, verbose=0):
        super().__init__(verbose)
        self.flag_rate_threshold = flag_rate_threshold
        self.window_size = window_size
        self.min_steps_per_phase = min_steps_per_phase
        self.per_level_min_rate = per_level_min_rate
        self.per_level_min_samples = per_level_min_samples
        self.campaign_mode = campaign_mode
        # Weighted-sampling rebalance: trigger every N episodes when imbalanced.
        self.rebalance_interval = rebalance_interval
        self._flag_window = deque(maxlen=window_size)
        # Per-level rolling history: {"1-1": deque([1,0,1,...]), ...}
        self._level_history = {}
        # Campaign-mode metrics
        self._campaign_clears = deque(maxlen=window_size)
        self._stages_per_run  = deque(maxlen=window_size)
        self._phase_start_steps = 0
        self.advance_requested   = False
        self.rebalance_requested = False
        self._episodes_since_rebalance = 0
        self._canonical_stages   = []  # set in reset_for_new_phase

    def update_threshold(self, new_threshold):
        self.flag_rate_threshold = new_threshold

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get('dones', []),
                              self.locals.get('infos', [])):
            if not done:
                continue
            flag = 1 if info.get('flag_get', False) else 0
            self._flag_window.append(flag)
            self._episodes_since_rebalance += 1

            if self.campaign_mode:
                self._campaign_clears.append(
                    1 if info.get('campaign_cleared', False) else 0)
                self._stages_per_run.append(info.get('stages_cleared', 0))
            else:
                level = f"{info.get('world', '?')}-{info.get('stage', '?')}"
                hist = self._level_history.setdefault(
                    level, deque(maxlen=self.window_size))
                hist.append(flag)

        steps_this_phase = self.num_timesteps - self._phase_start_steps

        if len(self._flag_window) == 0:
            return True

        # Prefer a uniform per-level average so that weighted sampling (which
        # overrepresents failing levels) doesn't suppress the metric below the
        # threshold for levels the agent has already mastered.
        # Fallback to the episode-window average when per-level data is sparse.
        per_level_rates_all = {
            lvl: sum(hist) / len(hist)
            for lvl, hist in self._level_history.items()
            if len(hist) >= self.per_level_min_samples
        }
        n_canonical = len(self._canonical_stages) if self._canonical_stages else 0
        if (not self.campaign_mode
                and n_canonical > 0
                and len(per_level_rates_all) >= n_canonical):
            # All canonical levels have enough samples: use uniform average.
            flag_rate = sum(per_level_rates_all.values()) / len(per_level_rates_all)
        else:
            flag_rate = sum(self._flag_window) / len(self._flag_window)

        self.logger.record('curriculum/flag_rate', flag_rate)
        self.logger.record('curriculum/flag_rate_target', self.flag_rate_threshold)
        self.logger.record('curriculum/episodes_tracked', len(self._flag_window))

        if self.campaign_mode and len(self._campaign_clears) > 0:
            campaign_rate = sum(self._campaign_clears) / len(self._campaign_clears)
            avg_stages    = sum(self._stages_per_run)  / len(self._stages_per_run)
            self.logger.record('campaign/clear_rate', campaign_rate)
            self.logger.record('campaign/avg_stages_per_run', avg_stages)

        # Log per-level rates and find weakest level (random-stage mode only).
        weak_level = None
        weak_rate  = 1.0
        if not self.campaign_mode:
            for level, hist in self._level_history.items():
                if len(hist) >= self.per_level_min_samples:
                    rate = sum(hist) / len(hist)
                    self.logger.record(f'curriculum/level_{level}', rate)
                    if rate < weak_rate:
                        weak_rate = rate
                        weak_level = level
            if weak_level is not None:
                self.logger.record('curriculum/weakest_level_rate', weak_rate)

        # ── Advance check (runs FIRST — takes priority over rebalance) ────────
        # If the agent is ready to advance, there's no point rebalancing.
        ready = (
            len(self._flag_window) >= self.window_size
            and steps_this_phase >= self.min_steps_per_phase
            and flag_rate >= self.flag_rate_threshold
        )
        if ready and self.per_level_min_rate > 0 and not self.campaign_mode:
            # Gate on the weakest level too — but only if we have enough
            # samples for every level in the pool. Otherwise the gate is
            # premature.
            ready = (weak_level is None) or (weak_rate >= self.per_level_min_rate)
            if not ready and weak_level is not None:
                # Periodically surface the bottleneck so it's visible without
                # tensorboard.
                self.logger.record('curriculum/blocking_level', 0)  # placeholder

        if ready:
            print(f"\n[ADVANCE] Curriculum advance! Flag rate {flag_rate:.1%} >= "
                  f"target {self.flag_rate_threshold:.0%} over "
                  f"{self.window_size} eps | {steps_this_phase/1e6:.1f}M steps this phase")
            if weak_level is not None:
                print(f"          (Weakest level was {weak_level} @ {weak_rate:.1%})")
            self.advance_requested = True
            return False

        # ── Rebalance check (random-stage mode only) ──────────────────────────
        # Every `rebalance_interval` episodes, if one level is clearly failing
        # while others are succeeding, request a venv rebuild with weighted
        # stage sampling. The main loop handles the actual rebuild.
        if (not self.campaign_mode
                and self._episodes_since_rebalance >= self.rebalance_interval
                and not self.rebalance_requested):
            level_rates = {
                lvl: sum(hist) / len(hist)
                for lvl, hist in self._level_history.items()
                if len(hist) >= self.per_level_min_samples
            }
            if level_rates:
                min_r = min(level_rates.values())
                max_r = max(level_rates.values())
                # Rebalance when there's a clear gap: best level ≥3× worst.
                if max_r >= 0.4 and min_r < max_r / 3:
                    print(f"\n[REBALANCE] Imbalance detected — "
                          f"best {max_r:.0%}, worst {min_r:.0%}. "
                          f"Rebuilding envs with weighted stage sampling.")
                    self.rebalance_requested = True
                    self._episodes_since_rebalance = 0
                    return False
            self._episodes_since_rebalance = 0  # reset counter even if no rebalance

        return True

    def reset_for_new_phase(self, current_steps, new_threshold,
                            per_level_min_rate=0.0, campaign_mode=False,
                            canonical_stages=None):
        self._phase_start_steps = current_steps
        self._flag_window.clear()
        self._level_history.clear()
        self._campaign_clears.clear()
        self._stages_per_run.clear()
        self.advance_requested   = False
        self.rebalance_requested = False
        self._episodes_since_rebalance = 0
        self.flag_rate_threshold = new_threshold
        self.per_level_min_rate  = per_level_min_rate
        self.campaign_mode       = campaign_mode
        self._canonical_stages   = list(canonical_stages) if canonical_stages else []


class ThrottleCallback(CheckpointCallback):
    """Checkpoint callback with optional thermal pause between iterations and storage limit."""
    def __init__(self, keep_last_k=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_last_k = keep_last_k

    def _on_rollout_end(self) -> None:
        if THROTTLE_PAUSE_S > 0:
            time.sleep(THROTTLE_PAUSE_S)
        return super()._on_rollout_end()
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        # Clean up old checkpoints to save storage
        if self.n_calls % self.save_freq == 0:
            import glob
            import os
            files = glob.glob(os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip"))
            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            if len(files) > self.keep_last_k:
                for f in files[:-self.keep_last_k]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        return result


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_env(env_id, stages, seed=0):
    def _init():
        # Each SubprocVecEnv worker runs in its own process. With 100+ workers
        # on a many-core cloud box, OpenCV's internal threading (used by
        # cvtColor/resize in MarioWrapper) oversubscribes the CPU and thrashes,
        # tanking rollout FPS. Pin every worker's OpenCV to a single thread.
        cv2.setNumThreads(1)
        env = gym_super_mario_bros.make(env_id, stages=stages, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = MarioReward(env)
        env = SkipFrame(env, skip=4)
        env = Monitor(env)
        return env
    return _init


def make_campaign_env(seed=0):
    """Full-game env: SuperMarioBros-v0 with shared 3-life pool.
    Single-stage env type returns done on death OR flag; multi-stage env
    (this one — no stage suffix) returns done only on game-over (out of
    lives). Stage transitions happen in-game automatically on flag.
    """
    def _init():
        cv2.setNumThreads(1)  # see make_env: avoid OpenCV thread oversubscription
        env = gym_super_mario_bros.make(
            'SuperMarioBros-v0', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = MarioCampaignReward(env)
        env = SkipFrame(env, skip=4)
        env = Monitor(env, info_keywords=('stages_cleared', 'campaign_cleared'))
        return env
    return _init


# 'fork' start method: workers are cloned from the already-initialized parent
# instead of re-importing train.py from scratch. The default (forkserver/spawn)
# makes every worker re-run module-level code — re-importing torch (~5s each)
# and querying CUDA — which made startup crawl for minutes and never reach the
# first rollout at high env counts. Workers only run CPU emulators and never
# touch CUDA, so forking after the parent's CUDA init is safe here.
def make_venv(env_id, stages, n_envs):
    env_fns = [make_env(env_id, stages, seed=i) for i in range(n_envs)]
    venv = SubprocVecEnv(env_fns, start_method='fork')
    return VecFrameStack(venv, n_stack=4, channels_order='last')


def make_campaign_venv(n_envs):
    env_fns = [make_campaign_env(seed=i) for i in range(n_envs)]
    venv = SubprocVecEnv(env_fns, start_method='fork')
    return VecFrameStack(venv, n_stack=4, channels_order='last')


def compute_weighted_stages(canonical_stages, level_history, min_samples=40, max_ratio=6):
    """
    Return an expanded stages list where failing levels appear more often.
    Used for adaptive rebalancing: when 1-3 is at 0% and 1-1 is at 90%, the
    uniform random draw gives 1-3 only ~5% of gradient steps. Overrepresenting
    failing levels pushes more attempts through the death point and gives the
    value function a chance to see late-level states.

    Weight for level k ∝ (1 - flag_rate_k + ε).
    Levels with < min_samples episodes get maximum weight (assume hard / unseen).
    max_ratio caps how dominant any single level can be relative to the easiest.

    The returned list is passed directly to the env as the stages pool;
    np.random.choice selects uniformly from it, so repetition = higher probability.
    """
    if not canonical_stages or len(canonical_stages) <= 1:
        return list(canonical_stages or [])

    EPSILON = 0.05
    raw = {}
    for lvl in canonical_stages:
        hist = level_history.get(lvl, deque())
        if len(hist) < min_samples:
            raw[lvl] = 1.0          # not enough data — treat as maximally hard
        else:
            flag_rate = sum(hist) / len(hist)
            raw[lvl] = max(EPSILON, 1.0 - flag_rate)

    min_w = min(raw.values())
    max_w = max(raw.values())
    # Compress range so no level is more than max_ratio × as frequent as easiest.
    # Linear rescale: min stays at min_w, max gets capped at min_w * max_ratio.
    if min_w > 0 and max_w / min_w > max_ratio:
        scale = min_w * (max_ratio - 1) / (max_w - min_w)
        raw = {lvl: min_w + (w - min_w) * scale for lvl, w in raw.items()}

    total = sum(raw.values())
    result = []
    for lvl in canonical_stages:
        # ×10 slots per unit weight gives ~10% resolution; floor at 1 slot.
        slots = max(1, round(raw[lvl] / total * len(canonical_stages) * 10))
        result.extend([lvl] * slots)
    return result


def get_latest_model(models_dir):
    # Prefer step-numbered checkpoints; parse the step number so the highest
    # step wins regardless of file system timestamps (copies, rsync, etc.).
    step_files = glob.glob(os.path.join(models_dir, 'mario_ppo_*_steps.zip'))
    if step_files:
        return max(step_files,
                   key=lambda f: int(os.path.basename(f).split('_')[2]))
    # Fall back to any zip (phase-final saves, etc.) sorted by mtime.
    files = glob.glob(os.path.join(models_dir, '*.zip'))
    return max(files, key=os.path.getmtime) if files else None


def load_curriculum_state(save_dir):
    """Persist curriculum phase and phase_start_steps across restarts.
    Returns (phase_idx, phase_start_steps) where phase_start_steps is -1
    if not persisted (fall back to model.num_timesteps on resume).
    """
    state_file = os.path.join(save_dir, 'curriculum_phase.txt')
    if os.path.exists(state_file):
        with open(state_file) as f:
            parts = f.read().strip().split()
            phase_idx = int(parts[0])
            phase_start_steps = int(parts[1]) if len(parts) > 1 else -1
            return phase_idx, phase_start_steps
    return 0, -1


def save_curriculum_state(save_dir, phase_idx, phase_start_steps):
    with open(os.path.join(save_dir, 'curriculum_phase.txt'), 'w') as f:
        f.write(f'{phase_idx} {phase_start_steps}')


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low-cpu', action='store_true',
                        help='Deadlock mode: 8 envs, 1 thread, 1.0s throttle pause.')
    parser.add_argument('--cloud', action='store_true',
                        help='Cloud mode: 96 envs, 8 threads, batch 2048. For 64+ vCPU Vast.ai instances.')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh: ignore all existing checkpoints, reset curriculum to Phase 0.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Explicit checkpoint path to resume from on the first phase '
                             '(overrides auto-detect). Used once, then auto-detect takes over.')
    args = parser.parse_args()

    if args.low_cpu:
        print(">>> MODE: LOW CPU (Deadlock mode)")
        n_envs, num_threads, throttle, batch_sz = 8, 1, 1.0, 256
    elif args.cloud:
        # IMPORTANT: SubprocVecEnv collects rollouts by stepping ALL envs
        # synchronously from one (GIL-bound) process, so throughput is bound by
        # that serial loop + per-step IPC, NOT by core count. Measured on a
        # 256-core box: 192 envs ran SLOWER than 24 envs locally (<600 vs
        # ~1400 FPS) because serializing 192 observations every step swamps the
        # main process. So we cap the env count low; the extra cores are unused
        # by design (a single synchronous vec-env cannot exploit them).
        cpus = os.cpu_count() or 64
        n_envs = 32
        num_threads = 8
        throttle = 0.0
        batch_sz = 512
        print(f">>> MODE: CLOUD ({cpus} CPUs detected → {n_envs} envs, "
              f"batch {batch_sz}, {num_threads} threads)")
    elif DEVICE == 'cuda':
        print(">>> MODE: GPU ACCELERATED (MAX PERFORMANCE)")
        # Pushing 24 envs to fully saturate the RTX 5080.
        n_envs, num_threads, throttle, batch_sz = 24, 4, 0.0, 512
    else:
        print(">>> MODE: FULL PERFORMANCE (CPU)")
        n_envs, num_threads, throttle, batch_sz = 16, 2, 0.5, 256

    # Collection is CPU-bound (single synchronous SubprocVecEnv loop), so the
    # GPU sits idle during it — an extra PPO epoch is nearly free in wall-clock
    # and improves how much the policy learns from each rollout (better sample
    # efficiency = fewer env steps to reach the goal). Still well below SB3's
    # default of 10, keeping the project's throughput-oriented tuning.
    n_epochs_cfg = 4

    global THROTTLE_PAUSE_S
    THROTTLE_PAUSE_S = throttle
    torch.set_num_threads(num_threads)

    env_id   = 'SuperMarioBrosRandomStages-v0'
    save_dir = './models_v2/'  # always use v2 dir; --fresh just skips loading on first run
    log_dir  = './logs/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.fresh:
        # Wipe curriculum state for the new run
        save_curriculum_state(save_dir, 0, 0)
        print("[FRESH START] Ignoring all existing checkpoints. Saving to models_v2/.")

    # Resume curriculum phase and step counter across restarts
    curriculum_idx, saved_phase_start_steps = load_curriculum_state(save_dir)
    _initial_stages = CURRICULUM_STAGES[curriculum_idx]
    _initial_label  = 'CAMPAIGN MODE (full game, 3 lives)' if _initial_stages is None else _initial_stages
    print(f"Starting at curriculum Phase {curriculum_idx + 1}/{len(CURRICULUM_STAGES)}: "
          f"{_initial_label} "
          f"(phase started at step {saved_phase_start_steps:,})")

    curriculum_cb = CurriculumCallback(
        flag_rate_threshold=FLAG_RATE_THRESHOLDS[curriculum_idx],
        window_size=FLAG_RATE_WINDOW,
        min_steps_per_phase=MIN_STEPS_PER_PHASE,
        per_level_min_rate=PER_LEVEL_MIN_RATES[curriculum_idx],
        campaign_mode=(CURRICULUM_STAGES[curriculum_idx] is None),
    )
    curriculum_cb._canonical_stages = list(CURRICULUM_STAGES[curriculum_idx] or [])

    resume_override = args.resume_from  # consumed on the first phase load only

    while curriculum_idx < len(CURRICULUM_STAGES):
        stages        = CURRICULUM_STAGES[curriculum_idx]
        phase_threshold = FLAG_RATE_THRESHOLDS[curriculum_idx]
        per_level_min   = PER_LEVEL_MIN_RATES[curriculum_idx]
        campaign_mode   = stages is None

        print(f"\n{'='*60}")
        if campaign_mode:
            print(f"Phase {curriculum_idx + 1}: CAMPAIGN MODE (full game, 3 lives)")
            print(f"Advance threshold : N/A (final phase — train indefinitely)")
        else:
            print(f"Phase {curriculum_idx + 1}: {stages}")
            print(f"Advance threshold : {phase_threshold:.0%} avg flag rate")
            if per_level_min > 0:
                print(f"Per-level minimum : {per_level_min:.0%} (weakest level must hit this)")
        print(f"Entropy coef      : {ENT_COEFS[curriculum_idx]}")
        print(f"{'='*60}")

        # active_stages is updated on each rebalance; canonical stages never change.
        active_stages = list(stages) if not campaign_mode else None

        # Build the initial venv and load the model once per phase.
        # Rebalance iterations rebuild the venv and swap via model.set_env(),
        # which works because every venv uses the same n_envs.
        _init_venv = make_campaign_venv(n_envs) if campaign_mode else make_venv(env_id, active_stages, n_envs)
        if resume_override:
            latest = resume_override
            resume_override = None  # only override the very first load
        elif args.fresh and curriculum_idx == 0:
            latest = None
        else:
            latest = get_latest_model(save_dir)
        if latest:
            print(f"Resuming from: {latest}")
            model = PPO.load(latest, env=_init_venv, device=DEVICE,
                             custom_objects={
                                 'learning_rate': 1.5e-4,
                                 'clip_range': 0.15,
                                 'batch_size': batch_sz,
                                 'n_epochs': n_epochs_cfg,
                                 'ent_coef': ENT_COEFS[curriculum_idx],
                             },
                             tensorboard_log=log_dir)
        else:
            print("Building new PPO model...")
            def linear_schedule(t): return t * 2.5e-4
            model = PPO('CnnPolicy', _init_venv, verbose=1,
                        tensorboard_log=log_dir,
                        learning_rate=linear_schedule,
                        ent_coef=ENT_COEFS[curriculum_idx],
                        n_steps=2048,
                        batch_size=batch_sz,
                        n_epochs=n_epochs_cfg,
                        gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, max_grad_norm=0.5, vf_coef=0.5,
                        device=DEVICE)

        # Use saved phase_start_steps on the first outer-loop iteration so
        # crash restarts don't reset the step counter. -1 = not persisted.
        effective_phase_start = (saved_phase_start_steps
                                 if saved_phase_start_steps >= 0
                                 else model.num_timesteps)
        saved_phase_start_steps = -1  # only use once
        curriculum_cb.reset_for_new_phase(
            effective_phase_start,
            new_threshold=FLAG_RATE_THRESHOLDS[curriculum_idx],
            per_level_min_rate=PER_LEVEL_MIN_RATES[curriculum_idx],
            campaign_mode=campaign_mode,
            canonical_stages=stages,
        )

        # ── Inner rebalance loop ───────────────────────────────────────────────
        # Runs the training loop for this phase. On rebalance the venv is
        # rebuilt with weighted stages and model.set_env() swaps it in without
        # reloading weights. On phase advance or keyboard interrupt, exits.
        venv = _init_venv  # first pass uses the venv already bound to the model
        while True:

            # Scale checkpoint frequency with n_envs so wall-clock interval
            # stays roughly constant. Local (24 envs): every 240k timesteps.
            # Cloud (192 envs): every ~1.9M timesteps. keep_last_k shrinks on
            # cloud to stay under ~300 MB.
            ckpt_freq = max(10000, 10000 * (n_envs // 24))
            ckpt_keep = 3 if args.cloud else 40
            checkpoint_cb = ThrottleCallback(
                keep_last_k=ckpt_keep,
                save_freq=ckpt_freq, save_path=save_dir, name_prefix='mario_ppo')
            callbacks = CallbackList([checkpoint_cb, curriculum_cb])

            try:
                model.learn(total_timesteps=100_000_000, callback=callbacks,
                            reset_num_timesteps=False)
            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving...")
                model.save(f"{save_dir}/mario_ppo_final")
                venv.close()
                return
            finally:
                model.save(f"{save_dir}/mario_ppo_phase{curriculum_idx + 1}_final")

            venv.close()

            if curriculum_cb.rebalance_requested:
                # Rebuild venv with weighted stages. Level history is preserved
                # so the next rebalance has fresh data from the new distribution.
                active_stages = compute_weighted_stages(
                    list(stages), curriculum_cb._level_history)
                curriculum_cb.rebalance_requested = False
                print(f"[REBALANCE] New stage pool ({len(active_stages)} slots): "
                      f"{active_stages}")
                venv = (make_campaign_venv(n_envs) if campaign_mode
                        else make_venv(env_id, active_stages, n_envs))
                model.set_env(venv)
                continue  # inner loop — train with reweighted venv

            break  # advance or interrupt — exit inner loop

        if curriculum_cb.advance_requested:
            curriculum_idx += 1
            save_curriculum_state(save_dir, curriculum_idx, model.num_timesteps)
            if curriculum_idx < len(CURRICULUM_STAGES):
                next_stages    = CURRICULUM_STAGES[curriculum_idx]
                next_threshold = FLAG_RATE_THRESHOLDS[curriculum_idx]
                next_label     = ('CAMPAIGN MODE (full game, 3 lives)'
                                  if next_stages is None else next_stages)
                if next_stages is None:
                    print(f"\n[PHASE UP] Advancing to Phase {curriculum_idx + 1}: "
                          f"{next_label}")
                else:
                    print(f"\n[PHASE UP] Advancing to Phase {curriculum_idx + 1}: "
                          f"{next_label} "
                          f"(target: {next_threshold:.0%})")
        else:
            print("Training stopped.")
            break

    print("\n[DONE] All curriculum phases complete!")


if __name__ == '__main__':
    main()
