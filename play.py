import os
import glob
import time
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import MarioWrapper, MarioReward, SkipFrame

# Fix pyglet 1.5.x bug on 64-bit Windows: HWND overflow
import sys
if sys.platform == 'win32':
    import ctypes
    from pyglet.libs.win32 import _user32
    _user32.CreateWindowExW.restype = ctypes.c_void_p
    _user32.GetDC.argtypes = [ctypes.c_void_p]
    _user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

def get_latest_model(models_dir):
    # Prefer step-numbered checkpoints over final/phase saves
    files = glob.glob(os.path.join(models_dir, 'mario_ppo_*_steps.zip'))
    if not files:
        files = glob.glob(os.path.join(models_dir, '*.zip'))
    return max(files, key=os.path.getctime) if files else None

def main():
    models_dir = './models_v2/'
    latest = get_latest_model(models_dir)

    if not latest:
        print("No trained models found in models_game/.")
        return

    print(f"Loading model: {latest}")
    model = PPO.load(latest, device='cpu')

    env_id = 'SuperMarioBros-v0'

    def make_env():
        env = gym_super_mario_bros.make(env_id, render_mode='human', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MarioWrapper(env)
        env = MarioReward(env)
        env = SkipFrame(env, skip=4)
        return env

    venv = DummyVecEnv([make_env])
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')

    obs = venv.reset()
    # Float window in Hyprland (silently fails if not Hyprland)
    os.system(f"hyprctl dispatch setfloating pid:{os.getpid()} >/dev/null 2>&1")
    os.system(f"hyprctl dispatch centerwindow pid:{os.getpid()} >/dev/null 2>&1")

    print("Evaluating! Ctrl+C to stop.")
    print("-" * 50)

    episode = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)  # stochastic — avoids locked-in failures
            obs, rewards, dones, info = venv.step(action)

            if dones[0]:
                episode += 1
                i = info[0]
                flag = "FLAG! 🎉" if i.get('flag_get', False) else "died"
                print(f"  Ep {episode}: {flag} | x={i.get('x_pos', 0)} | "
                      f"stage={i.get('world', '?')}-{i.get('stage', '?')}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")
    finally:
        venv.close()

if __name__ == '__main__':
    main()
