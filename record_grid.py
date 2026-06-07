"""
record_grid.py — Record a 5×3 tiled 16:9 video of 15 trained Mario agents.

Loads the latest PPO checkpoint and runs 15 environments simultaneously,
each on a different stage. 5 columns × 3 rows of 256×240 NES frames
gives exactly 1280×720 (16:9). Produces:
  • mario_grid_169.mp4  — ~25-30 s at 30 fps, H.264, 2× nearest-neighbour
  • mario_grid_169.png  — poster frame (a single representative frame)

Usage:
    python record_grid.py
    python record_grid.py --model models_v2/mario_ppo_latest.zip
    python record_grid.py --seconds 30 --scale 3
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # headless rendering

import argparse
import glob
import time
import numpy as np
import cv2
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import MarioWrapper, MarioReward, SkipFrame

# ─── STAGE POOL ──────────────────────────────────────────────────────────────
# 15 stages spread across all worlds for maximum visual variety.
GRID_STAGES = [
    '1-1', '2-1', '3-1', '4-1', '5-1',   # row 1
    '1-2', '2-3', '3-3', '4-2', '5-2',   # row 2
    '1-3', '1-4', '2-4', '3-4', '4-4',   # row 3
]

N_ENVS = 15
GRID_COLS = 5
GRID_ROWS = 3
FPS = 30

# NES native resolution
NES_H, NES_W = 240, 256


# ─── ENV FACTORY ─────────────────────────────────────────────────────────────

class MarioRGBCapture:
    """Thin pass-through that captures the raw RGB NES frame on every step.

    Must sit between JoypadSpace (old gym.Env) and MarioWrapper.
    Cannot inherit from gymnasium.Wrapper because JoypadSpace isn't a
    gymnasium.Env, so we delegate manually — same approach as MarioWrapper.
    """
    def __init__(self, env):
        self.env = env
        self.last_rgb = np.zeros((NES_H, NES_W, 3), dtype=np.uint8)
        # Expose spaces so downstream wrappers see them
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        result = self.env.step(action)
        self._capture()
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._capture()
        return result

    def _capture(self):
        frame = self.env.render()
        if frame is not None:
            self.last_rgb = frame

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, *args, **kwargs):
        if hasattr(self.env, 'seed'):
            return self.env.seed(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_grid_env(stage):
    """Build a single env with the same wrapper stack as training,
    plus an RGB capture wrapper for video recording."""
    world, lvl = stage.split('-')
    env_id = f'SuperMarioBros-{world}-{lvl}-v0'

    def _init():
        env = gym_super_mario_bros.make(
            env_id, render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # Insert RGB capture BEFORE the greyscale wrapper
        env = MarioRGBCapture(env)
        env = MarioWrapper(env)
        env = MarioReward(env)
        env = SkipFrame(env, skip=4)
        return env

    return _init


def extract_rgb_capture(env):
    """Walk the wrapper chain to find the MarioRGBCapture layer."""
    current = env
    while current is not None:
        if isinstance(current, MarioRGBCapture):
            return current
        current = getattr(current, 'env', None)
    return None


# ─── GRID ASSEMBLY ───────────────────────────────────────────────────────────

def assemble_grid(envs, scale=2):
    """Tile 15 RGB frames into a 5×3 grid (1280×720) and upscale.

    Uses np.hstack / np.vstack instead of SB3's auto-tiler, which would
    arrange 15 envs into a near-square layout with gaps.
    """
    black = np.zeros((NES_H, NES_W, 3), dtype=np.uint8)

    # Collect raw RGB frames from each env's MarioRGBCapture layer
    frames = []
    for env in envs:
        capture = extract_rgb_capture(env)
        frames.append(capture.last_rgb if capture is not None else black)

    # Build 3 rows of 5 tiles each, then stack vertically
    rows = []
    for r in range(GRID_ROWS):
        start = r * GRID_COLS
        row_frames = frames[start:start + GRID_COLS]
        # Pad if fewer than GRID_COLS (shouldn't happen with 15 envs)
        while len(row_frames) < GRID_COLS:
            row_frames.append(black)
        rows.append(np.hstack(row_frames))
    grid = np.vstack(rows)  # 720×1280×3

    if scale != 1:
        grid = cv2.resize(
            grid,
            (grid.shape[1] * scale, grid.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST)

    return grid


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def get_latest_model(models_dir='./models_v2'):
    step_files = glob.glob(os.path.join(models_dir, 'mario_ppo_*_steps.zip'))
    if step_files:
        return max(step_files,
                   key=lambda f: int(os.path.basename(f).split('_')[2]))
    files = glob.glob(os.path.join(models_dir, '*.zip'))
    return max(files, key=os.path.getmtime) if files else None


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Record a 5×3 (16:9) grid video of trained Mario agents.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model zip. Default: latest checkpoint.')
    parser.add_argument('--seconds', type=int, default=28,
                        help='Video duration in seconds (default: 28)')
    parser.add_argument('--scale', type=int, default=2,
                        help='Nearest-neighbour upscale factor (default: 2)')
    parser.add_argument('--output', type=str, default='mario_grid_169.mp4',
                        help='Output video path (default: mario_grid_169.mp4)')
    parser.add_argument('--poster', type=str, default='mario_grid_169.png',
                        help='Output poster frame path (default: mario_grid_169.png)')
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = args.model or get_latest_model()
    if not model_path:
        print('ERROR: No model found in models_v2/. '
              'Specify --model or place a checkpoint there.')
        return
    print(f'Model : {os.path.basename(model_path)}')
    model = PPO.load(model_path, device='cpu')

    # ── Build VecEnv ─────────────────────────────────────────────────────────
    stages = GRID_STAGES[:N_ENVS]
    print(f'Stages: {stages}')
    env_fns = [make_grid_env(s) for s in stages]
    venv = DummyVecEnv(env_fns)
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')

    # Get the underlying raw envs for RGB capture
    raw_envs = venv.venv.envs  # list of gymnasium envs (the inner DummyVecEnv)

    # ── Record loop ──────────────────────────────────────────────────────────
    total_frames = args.seconds * FPS
    # With skip=4, each model step covers 4 NES frames. We want 30 fps of
    # *video* output, so we record one video frame per model step.
    # At skip=4, that's ~7.5 NES fps — close enough for a showcase.

    print(f'\nRecording {args.seconds}s @ {FPS} fps → {total_frames} frames')
    print(f'Scale: {args.scale}× nearest-neighbour')

    import imageio
    writer = imageio.get_writer(
        args.output,
        fps=FPS,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        macro_block_size=1,  # allow odd resolutions
    )

    obs = venv.reset()
    poster_frame = None
    poster_saved = False
    t0 = time.time()

    for frame_idx in range(total_frames):
        # Stochastic actions for variety
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = venv.step(actions)
        # Auto-reset is handled by DummyVecEnv

        # Assemble the tiled grid from raw RGB captures
        grid = assemble_grid(raw_envs, scale=args.scale)
        writer.append_data(grid)

        # Capture poster frame ~2-3 seconds in (once gameplay is underway)
        if not poster_saved and frame_idx >= 2 * FPS:
            poster_frame = grid.copy()
            poster_saved = True

        # Progress
        if (frame_idx + 1) % (FPS * 5) == 0 or frame_idx == total_frames - 1:
            elapsed = time.time() - t0
            pct = (frame_idx + 1) / total_frames * 100
            print(f'  [{pct:5.1f}%] frame {frame_idx + 1}/{total_frames} '
                  f'({elapsed:.1f}s elapsed)')

    writer.close()
    venv.close()

    # ── Save poster PNG ──────────────────────────────────────────────────────
    if poster_frame is not None:
        # imageio writes RGB; save as-is
        imageio.imwrite(args.poster, poster_frame)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    vid_size = os.path.getsize(args.output) / (1024 * 1024)
    h, w = grid.shape[:2]
    from math import gcd
    g = gcd(w, h)
    aspect = f'{w // g}:{h // g}'
    print(f'\n{"=" * 55}')
    print(f'  Video      : {args.output}')
    print(f'  Poster     : {args.poster}')
    print(f'  Resolution : {w}×{h}  ({aspect})')
    print(f'  Duration   : {args.seconds}s @ {FPS} fps')
    print(f'  File size  : {vid_size:.1f} MB')
    print(f'  Time       : {elapsed:.1f}s')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
