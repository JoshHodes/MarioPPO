"""
Headless multi-level evaluation of the latest checkpoint.
Runs N episodes per level and reports completion rate, average x_pos, and reward.

Usage:
    python evaluate.py                    # All levels up to current curriculum phase
    python evaluate.py --levels 1-1 1-2   # Specific levels
    python evaluate.py --episodes 20      # Episodes per level (default: 10)
    python evaluate.py --deterministic    # Greedy policy (no exploration noise)
    python evaluate.py --campaign         # Evaluate full campaign (3 lives, 1-1 → 8-4)
    python evaluate.py --model path/to/model.zip
"""
import os
import glob
import argparse
import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import MarioWrapper, MarioReward, MarioCampaignReward, SkipFrame

# All levels up through W6 (held-out 6-1 excluded), in curriculum order.
ALL_LEVELS = [
    '1-1', '1-2', '1-3', '1-4',
    '2-1', '2-2', '2-3', '2-4',
    '3-1', '3-2', '3-3', '3-4',
    '4-1', '4-2', '4-3', '4-4',
    '5-1', '5-2', '5-3', '5-4',
    '6-2', '6-3', '6-4',
]


def get_latest_model(models_dir):
    files = glob.glob(os.path.join(models_dir, 'mario_ppo_*_steps.zip'))
    if files:
        return max(files, key=lambda f: int(os.path.basename(f).split('_')[2]))
    files = glob.glob(os.path.join(models_dir, '*.zip'))
    return max(files, key=os.path.getctime) if files else None


def make_env(level):
    world, stage = level.split('-')
    env_id = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioWrapper(env)
    env = MarioReward(env)
    env = SkipFrame(env, skip=4)
    return env


def make_campaign_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioWrapper(env)
    env = MarioCampaignReward(env)
    env = SkipFrame(env, skip=4)
    return env


def eval_campaign(model, n_episodes, deterministic):
    """Run full-game episodes (3 shared lives, 1-1 → 8-4) and report:
       - % episodes that beat 8-4
       - distribution of stages cleared per run
       - furthest world/stage reached
    """
    venv = DummyVecEnv([make_campaign_env])
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')

    runs = []  # list of (cleared_game, stages_cleared, furthest_world, furthest_stage, reward)
    obs = venv.reset()
    ep_reward = 0.0
    best_world, best_stage = 1, 1

    while len(runs) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, dones, info = venv.step(action)
        ep_reward += rew[0]
        i = info[0]
        w, s = i.get('world', 1), i.get('stage', 1)
        if (w, s) > (best_world, best_stage):
            best_world, best_stage = w, s
        if dones[0]:
            cleared = bool(i.get('campaign_cleared', False))
            stages  = int(i.get('stages_cleared', 0))
            runs.append((cleared, stages, best_world, best_stage, ep_reward))
            print(f"  Run {len(runs)}: cleared={cleared} | stages={stages} | "
                  f"furthest={best_world}-{best_stage} | reward={ep_reward:.0f}")
            ep_reward = 0.0
            best_world, best_stage = 1, 1
            obs = venv.reset()

    venv.close()
    return runs


def eval_level(model, level, n_episodes, deterministic):
    venv = DummyVecEnv([lambda l=level: make_env(l)])
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')

    flags, x_positions, rewards = [], [], []
    obs = venv.reset()

    ep_reward = 0.0
    while len(flags) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, dones, info = venv.step(action)
        ep_reward += rew[0]
        if dones[0]:
            i = info[0]
            flags.append(bool(i.get('flag_get', False)))
            x_positions.append(int(i.get('x_pos', 0)))
            rewards.append(ep_reward)
            ep_reward = 0.0
            obs = venv.reset()

    venv.close()
    return flags, x_positions, rewards


def bar(rate, width=20):
    filled = int(rate * width)
    return '█' * filled + '░' * (width - filled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', nargs='+', default=None,
                        help='Levels to evaluate (e.g. 1-1 1-2 2-1). Default: all.')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per level (default: 10)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use greedy policy instead of stochastic')
    parser.add_argument('--campaign', action='store_true',
                        help='Full-game evaluation: 3 shared lives, 1-1 → 8-4.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model zip. Default: latest checkpoint.')
    args = parser.parse_args()

    model_path = args.model or get_latest_model('./models_v2')
    if not model_path:
        print('No model found.')
        return

    mode = 'deterministic' if args.deterministic else 'stochastic'
    print(f'\nModel : {os.path.basename(model_path)}')
    print(f'Mode  : {mode}')

    model = PPO.load(model_path, device='cpu')

    if args.campaign:
        print(f'Eval  : CAMPAIGN ({args.episodes} runs)\n')
        runs = eval_campaign(model, args.episodes, args.deterministic)
        cleared = sum(1 for r in runs if r[0])
        avg_stages = np.mean([r[1] for r in runs])
        max_stages = max(r[1] for r in runs)
        # Furthest reached across all runs
        furthest = max((r[2], r[3]) for r in runs)
        print('─' * 62)
        print(f'  Full-game clear rate : {cleared}/{len(runs)} ({cleared/len(runs):.0%})')
        print(f'  Avg stages / run     : {avg_stages:.1f}')
        print(f'  Best run             : {max_stages} stages')
        print(f'  Furthest reached     : {furthest[0]}-{furthest[1]}')
        print()
        return

    levels = args.levels or ALL_LEVELS
    print(f'Episodes/level: {args.episodes}')
    print(f'Levels: {len(levels)}\n')

    print(f'{"Level":<8} {"Flag%":>6}  {"Completion":22}  {"Avg x":>6}  {"Avg reward":>10}')
    print('─' * 62)

    totals_flag = []
    for level in levels:
        flags, xs, rews = eval_level(model, level, args.episodes, args.deterministic)
        rate    = np.mean(flags)
        avg_x   = np.mean(xs)
        avg_rew = np.mean(rews)
        totals_flag.append(rate)

        indicator = '✓' if rate >= 0.5 else ('~' if rate >= 0.2 else '✗')
        print(f'{level:<8} {rate:>5.0%}  {bar(rate):22}  {avg_x:>6.0f}  {avg_rew:>10.1f}  {indicator}')

    print('─' * 62)
    overall = np.mean(totals_flag)
    print(f'{"OVERALL":<8} {overall:>5.0%}  {bar(overall):22}\n')


if __name__ == '__main__':
    main()
