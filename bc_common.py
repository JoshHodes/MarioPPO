"""
bc_common.py — shared pieces for the behavioral-cloning (imitation-learning)
experiment.

The idea: PPO from scratch hit a hard-exploration wall (1-3 stuck at 0% even
with RND curiosity). Instead of making the agent *discover* the level, we give
it a teacher (human demonstrations), train a reactive CNN to *imitate* the
demonstrated pixel->action mapping (behavioral cloning), and check whether that
clears the level PPO never could. The result is still a learned policy: at
inference it sees only stacked grayscale frames and outputs a button press, with
no emulator look-ahead.

This module holds the bits the recorder (`record_demos.py`) and the trainer
(`behavioral_clone.py`) both need, so the observation pipeline is IDENTICAL in
recording, training, and evaluation:
  - the env construction (single-level, SIMPLE_MOVEMENT, raw RGB obs),
  - frame preprocessing (grayscale -> 84x84) matching train.MarioWrapper,
  - 4-frame stacking + frameskip-4 (matching train's SkipFrame + VecFrameStack),
  - the policy network (same NatureCNN trunk as the PPO CnnPolicy),
  - a headless rollout evaluator (flag-rate on a level).
"""
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

NUM_ACTIONS = len(SIMPLE_MOVEMENT)   # 7
N_STACK = 4
SKIP = 4                              # frameskip — must match train.SkipFrame
cv2.setNumThreads(1)


# ─── env + observation pipeline ───────────────────────────────────────────────
def make_level_env(level='1-3'):
    """Single-level env returning RAW RGB frames (240,256,3). We grayscale/resize
    ourselves so the same rgb frame feeds both the on-screen display and the
    training observation. Matches train.make_env's emulator + action space."""
    env_id = f'SuperMarioBros-{level}-v0'
    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


def reset_env(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def step_env(env, action):
    """Normalize the gym/gymnasium 4-vs-5-tuple step API to (obs,r,done,info)."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, reward, done, info


def process_frame(rgb):
    """RGB (240,256,3) -> grayscale 84x84 uint8. Mirrors train.MarioWrapper."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


class FrameStacker:
    """Holds the last N_STACK processed frames, oldest-first, newest last —
    consistent across record/train/eval (the only thing that matters for a
    from-scratch BC net)."""
    def __init__(self, n=N_STACK):
        self.n = n
        self.buf = collections.deque(maxlen=n)

    def reset(self, frame):
        for _ in range(self.n):
            self.buf.append(frame)
        return self.get()

    def push(self, frame):
        self.buf.append(frame)
        return self.get()

    def get(self):
        # (84,84,N) uint8, newest in the last channel
        return np.stack(self.buf, axis=-1)


def obs_to_tensor(obs_hwc, device):
    """(84,84,N) uint8 -> (1,N,84,84) float in [0,1] for the CNN."""
    x = np.transpose(obs_hwc.astype(np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0).to(device)


# ─── policy network (same trunk as SB3's NatureCNN / CnnPolicy) ────────────────
class BCPolicy(nn.Module):
    def __init__(self, n_stack=N_STACK, n_actions=NUM_ACTIONS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):                 # x: (B,n_stack,84,84) float in [0,1]
        return self.head(self.cnn(x))     # logits (B,n_actions)


# ─── action mapping for the human recorder ─────────────────────────────────────
def keys_to_action(left, right, jump, run):
    """Map held buttons to the nearest SIMPLE_MOVEMENT index.
       0 NOOP | 1 right | 2 right+A | 3 right+B | 4 right+A+B | 5 A | 6 left"""
    if right and jump and run:
        return 4
    if right and jump:
        return 2
    if right and run:
        return 3
    if right:
        return 1
    if left:
        return 6          # SIMPLE_MOVEMENT has no left+A, so jump is dropped
    if jump:
        return 5
    return 0


# ─── headless rollout evaluator ────────────────────────────────────────────────
@torch.no_grad()
def rollout(policy, level='1-3', episodes=20, device='cpu',
            deterministic=True, max_steps=4000):
    """Run the BC policy on a level, frameskip-4 + 4-stack, no rendering.
    Returns (flag_rate, mean_max_x, per_episode list of (cleared, max_x))."""
    policy.eval()
    env = make_level_env(level)
    results = []
    for _ in range(episodes):
        stacker = FrameStacker()
        obs = stacker.reset(process_frame(reset_env(env)))
        cleared, max_x = False, 0
        for _ in range(max_steps):
            logits = policy(obs_to_tensor(obs, device))
            if deterministic:
                action = int(logits.argmax(dim=1).item())
            else:
                probs = torch.softmax(logits, dim=1)
                action = int(torch.multinomial(probs, 1).item())

            done = False
            for _ in range(SKIP):          # frameskip
                rgb, _, done, info = step_env(env, action)
                if done:
                    break
            obs = stacker.push(process_frame(rgb))
            max_x = max(max_x, int(info.get('x_pos', 0)))
            if info.get('flag_get', False):
                cleared = True
            if done:
                break
        results.append((cleared, max_x))
    env.close()
    flag_rate = np.mean([c for c, _ in results]) if results else 0.0
    mean_max_x = np.mean([x for _, x in results]) if results else 0.0
    return float(flag_rate), float(mean_max_x), results
