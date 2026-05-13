import os
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
    # Phase 2 — Worlds 1-2
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4'],
    # Phase 3 — Worlds 1-4
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4'],
    # Phase 4 — All 8 worlds (endgame)
    ['1-1', '1-2', '1-3', '1-4',
     '2-1', '2-2', '2-3', '2-4',
     '3-1', '3-2', '3-3', '3-4',
     '4-1', '4-2', '4-3', '4-4',
     '5-1', '5-2', '5-3', '5-4',
     '6-1', '6-2', '6-3', '6-4',
     '7-1', '7-2', '7-3', '7-4',
     '8-1', '8-2', '8-3', '8-4'],
]

# Per-phase flag rate thresholds.
FLAG_RATE_THRESHOLDS = [
    0.80,  # Phase 0: 1-1 only     (easy level, high bar)
    0.55,  # Phase 1: World 1      (require solid mastery before adding water levels)
    0.50,  # Phase 2: Worlds 1-2   (underwater + 2 castles)
    0.45,  # Phase 3: Worlds 1-4   (4 castles + 2 water levels)
    0.40,  # Phase 4: All 8 worlds (endgame)
]
# Per-phase entropy coefficients.
ENT_COEFS = [
    0.03,  # Phase 0: 1-1 only     (low — one easy level, exploit the flag signal)
    0.05,  # Phase 1: World 1      (slight boost for harder levels)
    0.08,  # Phase 2: Worlds 1-2   (boost to discover swimming)
    0.05,  # Phase 3: Worlds 1-4   (moderate — some water already learned)
    0.03,  # Phase 4: All 8 worlds (tighten up for endgame)
]
FLAG_RATE_WINDOW    = 200    # rolling window: number of completed episodes
MIN_STEPS_PER_PHASE = 5_000_000  # guard against advancing before window fills; flag rate is the real gate

# ─── REWARD ───────────────────────────────────────────────────────────────────

class MarioReward(gym.Wrapper):
    """
    Game-beating reward signal (adapted from successful GitHub implementations):
      - Forward progress: +0.1 per pixel of x movement.
      - Score delta: +1.0 per 40 points (gives +2.5 for a Goomba kill). Teaches enemy stomping naturally.
      - Death penalty: -50.
      - Flag bonus: +50 (reduced from 300 to avoid destabilizing gradients).
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0
        self.curr_score = 0

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
        if done:
            if info.get('flag_get', False):
                shaped_reward += 50.0
            else:
                shaped_reward -= 50.0

        if len(result) == 5:
            return obs, shaped_reward, terminated, truncated, info
        return obs, shaped_reward, done, info

    def reset(self, **kwargs):
        self.prev_x = 0
        self.curr_score = 0
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
    Tracks flag-completion rate over a rolling window of episodes.
    When rate >= flag_rate_threshold (after MIN_STEPS_PER_PHASE steps),
    sets advance_requested=True and stops training so the main loop
    can recreate the environment with the next curriculum phase.
    Threshold is set per-phase via update_threshold().
    """
    def __init__(self, flag_rate_threshold, window_size, min_steps_per_phase, verbose=0):
        super().__init__(verbose)
        self.flag_rate_threshold = flag_rate_threshold
        self.window_size = window_size
        self.min_steps_per_phase = min_steps_per_phase
        self._flag_window = deque(maxlen=window_size)
        self._phase_start_steps = 0
        self.advance_requested = False

    def update_threshold(self, new_threshold):
        """Call this when advancing to a new phase to set the new target."""
        self.flag_rate_threshold = new_threshold

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done:
                self._flag_window.append(1 if info.get('flag_get', False) else 0)

        steps_this_phase = self.num_timesteps - self._phase_start_steps
        if len(self._flag_window) > 0:
            flag_rate = sum(self._flag_window) / len(self._flag_window)
            self.logger.record('curriculum/flag_rate', flag_rate)
            self.logger.record('curriculum/flag_rate_target', self.flag_rate_threshold)
            self.logger.record('curriculum/episodes_tracked', len(self._flag_window))

            if (len(self._flag_window) >= self.window_size
                    and steps_this_phase >= self.min_steps_per_phase
                    and flag_rate >= self.flag_rate_threshold):
                print(f"\n[ADVANCE] Curriculum advance! Flag rate {flag_rate:.1%} >= "
                      f"target {self.flag_rate_threshold:.0%} over "
                      f"{self.window_size} eps | {steps_this_phase/1e6:.1f}M steps this phase")
                self.advance_requested = True
                return False  # stop learn() loop

        return True

    def reset_for_new_phase(self, current_steps, new_threshold):
        self._phase_start_steps = current_steps
        self._flag_window.clear()
        self.advance_requested = False
        self.flag_rate_threshold = new_threshold


class ThrottleCallback(CheckpointCallback):
    """Checkpoint callback with optional thermal pause between iterations."""
    def _on_rollout_end(self) -> None:
        if THROTTLE_PAUSE_S > 0:
            time.sleep(THROTTLE_PAUSE_S)
        return super()._on_rollout_end()


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_env(env_id, stages, seed=0):
    def _init():
        env = gym_super_mario_bros.make(env_id, stages=stages, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = MarioReward(env)
        env = SkipFrame(env, skip=4)
        env = Monitor(env)
        return env
    return _init


def make_venv(env_id, stages, n_envs):
    env_fns = [make_env(env_id, stages, seed=i) for i in range(n_envs)]
    venv = SubprocVecEnv(env_fns)
    return VecFrameStack(venv, n_stack=4, channels_order='last')


def get_latest_model(models_dir):
    files = glob.glob(os.path.join(models_dir, '*.zip'))
    return max(files, key=os.path.getctime) if files else None


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
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh: ignore all existing checkpoints, reset curriculum to Phase 0.')
    args = parser.parse_args()

    if args.low_cpu:
        print(">>> MODE: LOW CPU (Deadlock mode)")
        n_envs, num_threads, throttle = 8, 1, 1.0
    elif DEVICE == 'cuda':
        print(">>> MODE: GPU ACCELERATED (MAX PERFORMANCE)")
        # Pushing 24 envs to fully saturate the RTX 5080.
        n_envs, num_threads, throttle = 24, 4, 0.0
    else:
        print(">>> MODE: FULL PERFORMANCE (CPU)")
        n_envs, num_threads, throttle = 16, 2, 0.5

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
    print(f"Starting at curriculum Phase {curriculum_idx + 1}/{len(CURRICULUM_STAGES)}: "
          f"{CURRICULUM_STAGES[curriculum_idx]} "
          f"(phase started at step {saved_phase_start_steps:,})")

    curriculum_cb = CurriculumCallback(
        flag_rate_threshold=FLAG_RATE_THRESHOLDS[curriculum_idx],
        window_size=FLAG_RATE_WINDOW,
        min_steps_per_phase=MIN_STEPS_PER_PHASE,
    )

    while curriculum_idx < len(CURRICULUM_STAGES):
        stages = CURRICULUM_STAGES[curriculum_idx]
        phase_threshold = FLAG_RATE_THRESHOLDS[curriculum_idx]
        print(f"\n{'='*60}")
        print(f"Phase {curriculum_idx + 1}: {stages}")
        print(f"Advance threshold : {phase_threshold:.0%} flag rate")
        print(f"Entropy coef      : {ENT_COEFS[curriculum_idx]}")
        print(f"{'='*60}")

        venv = make_venv(env_id, stages, n_envs)

        latest = None if args.fresh and curriculum_idx == 0 else get_latest_model(save_dir)
        if latest:
            print(f"Resuming from: {latest}")
            model = PPO.load(latest, env=venv, device=DEVICE,
                             custom_objects={
                                 'learning_rate': 1.5e-4,   # reduced from 2.5e-4 — high clip_fraction caused collapse
                                 'clip_range': 0.15,        # tightened from 0.2 — prevents over-large policy updates
                                 'batch_size': 512 if DEVICE == 'cuda' else 256,
                                 'n_epochs': 3,
                                 'ent_coef': ENT_COEFS[curriculum_idx],
                             },
                             tensorboard_log=log_dir)
        else:
            print("Building new PPO model...")
            def linear_schedule(t): return t * 2.5e-4
            model = PPO('CnnPolicy', venv, verbose=1,
                        tensorboard_log=log_dir,
                        learning_rate=linear_schedule,
                        ent_coef=ENT_COEFS[curriculum_idx],
                        n_steps=2048,
                        batch_size=512 if DEVICE == 'cuda' else 256,
                        n_epochs=3,
                        gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, max_grad_norm=0.5, vf_coef=0.5,
                        device=DEVICE)

        checkpoint_cb = ThrottleCallback(
            save_freq=10000, save_path=save_dir, name_prefix='mario_ppo')
        # Use saved phase_start_steps on first loop iteration so crash restarts
        # don't reset the step counter back to the checkpoint step.
        # -1 means "not persisted" — fall back to model.num_timesteps.
        # 0 is valid and means training started from scratch (step 0).
        effective_phase_start = saved_phase_start_steps if saved_phase_start_steps >= 0 else model.num_timesteps
        saved_phase_start_steps = -1  # only use saved value on the first iteration
        curriculum_cb.reset_for_new_phase(
            effective_phase_start,
            new_threshold=FLAG_RATE_THRESHOLDS[curriculum_idx],
        )

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
            # Always save on phase exit
            model.save(f"{save_dir}/mario_ppo_phase{curriculum_idx + 1}_final")

        venv.close()

        if curriculum_cb.advance_requested:
            curriculum_idx += 1
            save_curriculum_state(save_dir, curriculum_idx, model.num_timesteps)
            if curriculum_idx < len(CURRICULUM_STAGES):
                next_threshold = FLAG_RATE_THRESHOLDS[curriculum_idx]
                print(f"\n[PHASE UP] Advancing to Phase {curriculum_idx + 1}: "
                      f"{CURRICULUM_STAGES[curriculum_idx]} "
                      f"(target: {next_threshold:.0%})")
        else:
            # Interrupted without triggering advance — stop
            print("Training stopped.")
            break

    print("\n[DONE] All curriculum phases complete!")


if __name__ == '__main__':
    main()
