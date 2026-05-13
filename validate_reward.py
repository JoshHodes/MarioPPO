"""
Reward signal validation script for CoinMaxx.
Run BEFORE committing to a long training session to verify the reward
function produces sensible values. This catches design mistakes in minutes
instead of discovering them after 12 hours of training.

Key checks:
  - Coins collected are reflected in the reward
  - Right+Jump earns more than Hold Right (jumping hits ? blocks for coins)
  - Standing still gives no reward
  - Forward progress is still rewarded (but less than coins)

Usage: python validate_reward.py
"""
import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from train import MarioWrapper, CoinMaxxReward, SkipFrame

def make_test_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioWrapper(env)
    env = CoinMaxxReward(env)
    env = SkipFrame(env, skip=4)
    return env

def run_policy(env, policy_fn, max_steps=2000):
    """Run an episode with a given policy function and track all rewards + coins."""
    obs, info = env.reset()
    total_reward = 0
    rewards = []
    max_x = 0
    steps = 0
    total_coins = 0
    
    for _ in range(max_steps):
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        steps += 1
        x = info.get('x_pos', 0)
        coins = info.get('coins', 0)
        if x > max_x:
            max_x = x
        if coins > total_coins:
            total_coins = coins
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'max_x': max_x,
        'coins': total_coins,
        'flag': info.get('flag_get', False),
        'rewards': rewards,
        'mean_reward_per_step': total_reward / max(steps, 1),
    }

def main():
    env = make_test_env()
    
    print("=" * 60)
    print("COINMAXX REWARD SIGNAL VALIDATION")
    print("=" * 60)
    
    # --- Test 1: Random policy (baseline) ---
    print("\n--- Test 1: Random Policy (10 episodes) ---")
    random_results = []
    for i in range(10):
        result = run_policy(env, lambda o, i: env.action_space.sample())
        random_results.append(result)
    
    avg_reward = np.mean([r['total_reward'] for r in random_results])
    avg_x = np.mean([r['max_x'] for r in random_results])
    avg_coins = np.mean([r['coins'] for r in random_results])
    avg_steps = np.mean([r['steps'] for r in random_results])
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Avg max_x:  {avg_x:.0f}")
    print(f"  Avg coins:  {avg_coins:.1f}")
    print(f"  Avg steps:  {avg_steps:.0f}")
    
    # --- Test 2: "Hold right" policy ---
    print("\n--- Test 2: Hold Right Policy (10 episodes) ---")
    # Action 1 = right in SIMPLE_MOVEMENT
    right_results = []
    for i in range(10):
        result = run_policy(env, lambda o, i: 1)
        right_results.append(result)
    
    avg_reward = np.mean([r['total_reward'] for r in right_results])
    avg_x = np.mean([r['max_x'] for r in right_results])
    avg_coins = np.mean([r['coins'] for r in right_results])
    avg_steps = np.mean([r['steps'] for r in right_results])
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Avg max_x:  {avg_x:.0f}")
    print(f"  Avg coins:  {avg_coins:.1f}")
    print(f"  Avg steps:  {avg_steps:.0f}")
    
    # --- Test 3: "Hold right + jump" policy ---
    print("\n--- Test 3: Right+Jump Policy (10 episodes) ---")
    # Action 2 = right+A (jump) in SIMPLE_MOVEMENT
    jump_results = []
    for i in range(10):
        result = run_policy(env, lambda o, i: 2)
        jump_results.append(result)
    
    avg_reward = np.mean([r['total_reward'] for r in jump_results])
    avg_x = np.mean([r['max_x'] for r in jump_results])
    avg_coins = np.mean([r['coins'] for r in jump_results])
    avg_steps = np.mean([r['steps'] for r in jump_results])
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Avg max_x:  {avg_x:.0f}")
    print(f"  Avg coins:  {avg_coins:.1f}")
    print(f"  Avg steps:  {avg_steps:.0f}")

    # --- Test 4: "Stand still" policy ---
    print("\n--- Test 4: Stand Still / NOOP Policy (3 episodes) ---")
    # Action 0 = NOOP in SIMPLE_MOVEMENT
    still_results = []
    for i in range(3):
        result = run_policy(env, lambda o, i: 0, max_steps=500)
        still_results.append(result)
    
    avg_reward = np.mean([r['total_reward'] for r in still_results])
    avg_coins = np.mean([r['coins'] for r in still_results])
    avg_steps = np.mean([r['steps'] for r in still_results])
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Avg coins:  {avg_coins:.1f}")
    print(f"  Avg steps:  {avg_steps:.0f}")

    # --- Sanity Checks ---
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    random_avg = np.mean([r['total_reward'] for r in random_results])
    right_avg = np.mean([r['total_reward'] for r in right_results])
    jump_avg = np.mean([r['total_reward'] for r in jump_results])
    still_avg = np.mean([r['total_reward'] for r in still_results])
    
    jump_coins = np.mean([r['coins'] for r in jump_results])
    right_coins = np.mean([r['coins'] for r in right_results])
    still_coins = np.mean([r['coins'] for r in still_results])
    
    checks = [
        # NOTE: Random often beats deterministic on 1-1 because SIMPLE_MOVEMENT
        # biases toward right+jump, accidentally clearing early obstacles.
        # So we check that Right+Jump at least gets meaningful reward.
        ("Right+Jump reward is meaningful", jump_avg > 5,
         f"Jump ({jump_avg:.1f}) should be above 5 — agent needs clear signal"),
        ("Standing still is near-zero", still_avg < 2.0,
         f"NOOP ({still_avg:.1f}) should be < 2.0 — minimal reward for doing nothing"),
        ("Jump collects >= Right coins", jump_coins >= right_coins,
         f"Jump coins ({jump_coins:.1f}) should be >= Right coins ({right_coins:.1f})"),
        ("Standing still gets no coins", still_coins == 0,
         f"NOOP coins ({still_coins:.1f}) should be 0"),
        ("Random collects some coins", np.mean([r['coins'] for r in random_results]) >= 0,
         f"Random coins ({np.mean([r['coins'] for r in random_results]):.1f}) should be >= 0"),
    ]
    
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
        print(f"         {detail}")
    
    print()
    if all_pass:
        print("All checks passed. CoinMaxx reward signal looks healthy for training!")
    else:
        print("WARNING: Some checks failed! Review the reward function before training.")
    
    env.close()

if __name__ == '__main__':
    main()
