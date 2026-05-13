"""
Quick test to verify the CoinMaxxReward wrapper tracks coins correctly.
Runs a few random steps and prints the coin count + shaped reward.

Usage: python test_agent.py
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from train import MarioWrapper, CoinMaxxReward, SkipFrame

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioWrapper(env)
    env = CoinMaxxReward(env)
    env = SkipFrame(env, skip=4)
    
    obs, info = env.reset()
    total_reward = 0
    prev_coins = 0
    
    print("Running 500 random steps on 1-1...")
    print(f"{'Step':>5} {'Action':>6} {'Reward':>8} {'Total':>8} {'Coins':>5} {'x_pos':>6}")
    print("-" * 50)
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        coins = info.get('coins', 0)
        # Only print when something interesting happens
        if abs(reward) > 1.0 or coins != prev_coins:
            print(f"{step:5d} {action:6d} {reward:8.2f} {total_reward:8.2f} {coins:5d} {info.get('x_pos', 0):6d}")
        prev_coins = coins
        
        if terminated or truncated:
            print(f"\n--- Episode ended at step {step} ---")
            print(f"Final coins: {coins} | Final x_pos: {info.get('x_pos', 0)}")
            print(f"Flag: {info.get('flag_get', False)} | Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            prev_coins = 0
            print("\n--- New episode ---")
    
    env.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
