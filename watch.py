"""
watch.py — Live render of the Mario agent as it trains.

Loads the latest checkpoint from models_v2/, plays one episode rendered to
screen, then automatically picks up any newer checkpoint before the next
episode. Zero impact on training — runs completely independently.

Usage:
    python watch.py                  # random stage from current curriculum phase
    python watch.py --stage 1-2      # always watch a specific stage
    python watch.py --deterministic  # greedy policy (no exploration noise)
"""

import os
import glob
import time
import argparse
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import MarioWrapper, MarioReward, SkipFrame, CURRICULUM_STAGES, load_curriculum_state

# Fix pyglet 1.5.x bug on 64-bit Windows: HWND overflow
import sys
if sys.platform == 'win32':
    import ctypes
    from pyglet.libs.win32 import _user32
    _user32.CreateWindowExW.restype  = ctypes.c_void_p
    _user32.GetDC.argtypes           = [ctypes.c_void_p]
    _user32.ReleaseDC.argtypes       = [ctypes.c_void_p, ctypes.c_void_p]

MODELS_DIR = './models_v2/'

def extract_nes_env(env):
    if hasattr(env, 'viewer'): return env
    if hasattr(env, 'env'): return extract_nes_env(env.env)
    if hasattr(env, 'envs'): return extract_nes_env(env.envs[0])
    if hasattr(env, 'venv'): return extract_nes_env(env.venv)
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'viewer'): return env.unwrapped
    return env

def get_latest_model():
    files = glob.glob(os.path.join(MODELS_DIR, 'mario_ppo_*_steps.zip'))
    return max(files, key=os.path.getctime) if files else None


def make_env(stage: str):
    """Single rendered env for a specific stage."""
    # SuperMarioBros-v0 for a fixed stage, with human render
    if '-' in stage:
        world, lvl = stage.split('-')
        env_id = f'SuperMarioBros-{world}-{lvl}-v0'
    else:
        env_id = 'SuperMarioBros-v0'

    def _init():
        env = gym_super_mario_bros.make(env_id, render_mode='human', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = MarioReward(env)
        env = SkipFrame(env, skip=4)
        return env

    venv = DummyVecEnv([_init])
    return VecFrameStack(venv, n_stack=4, channels_order='last')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default=None,
                        help='Fix a stage to watch, e.g. "1-2". Default: cycles through current phase stages.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use greedy (deterministic) policy. Default: stochastic.')
    args = parser.parse_args()

    latest_path = None
    model       = None
    venv        = None
    episode     = 0
    stage_idx   = 0
    current_stage = None

    print("=" * 55)
    print("  Mario Live Watcher — Ctrl+C to stop")
    print("=" * 55)

    try:
        while True:
            # ── Pick up newer checkpoint if available ────────────────
            new_path = get_latest_model()
            if new_path and new_path != latest_path:
                print(f"\n[LOAD] {os.path.basename(new_path)}")
                model       = PPO.load(new_path, device='cpu')
                latest_path = new_path

            if model is None:
                print("Waiting for a checkpoint in models_v2/ ...")
                time.sleep(5)
                continue

            # ── Pick stage ───────────────────────────────────────────
            if args.stage:
                stage = args.stage
            else:
                phase_idx, _ = load_curriculum_state(MODELS_DIR)
                phase_stages  = CURRICULUM_STAGES[phase_idx]
                stage         = phase_stages[stage_idx % len(phase_stages)]
                stage_idx    += 1

            # ── Build env for this episode ───────────────────────────
            if venv is None:
                venv = make_env(stage)
                current_stage = stage
                obs = venv.reset()
                # Float window in Hyprland (silently fails if not Hyprland)
                os.system(f"hyprctl dispatch setfloating pid:{os.getpid()} >/dev/null 2>&1")
                os.system(f"hyprctl dispatch centerwindow pid:{os.getpid()} >/dev/null 2>&1")
            elif current_stage != stage:
                # To prevent piling up windows, transfer the ImageViewer to the new env
                nes = extract_nes_env(venv)
                viewer = nes.viewer
                nes.viewer = None
                venv.close()
                
                venv = make_env(stage)
                nes_new = extract_nes_env(venv)
                nes_new.viewer = viewer
                
                current_stage = stage
                obs = venv.reset()
            else:
                obs = venv.reset()

            print(f"\n  Episode {episode + 1} | Stage {stage} | "
                  f"{'Greedy' if args.deterministic else 'Stochastic'} | "
                  f"{os.path.basename(latest_path)}")

            # ── Play one episode ─────────────────────────────────────
            total_reward = 0.0
            steps        = 0
            while True:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = venv.step(action)
                total_reward += float(reward[0])
                steps        += 1
                time.sleep(0.016)   # ~60 fps display

                if done[0]:
                    episode += 1
                    i        = info[0]
                    flag     = "🎉 FLAG!" if i.get('flag_get', False) else "💀 died"
                    print(f"  {flag} | x={i.get('x_pos', 0):4d} | "
                          f"reward={total_reward:6.1f} | steps={steps}")
                    break

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")
    finally:
        if venv is not None:
            venv.close()


if __name__ == '__main__':
    main()
