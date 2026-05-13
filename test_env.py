import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def main():
    # 1. Initialize the environment
    # 'SuperMarioBros-v0' is the standard level 1-1 with standard graphics
    # apply_api_compatibility=True ensures it works with newer versions of Gym
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    
    # 2. Limit the button presses
    # By default, the AI can press any combination of the 8 NES buttons (256 combinations)
    # SIMPLE_MOVEMENT reduces this to just 7 basic actions (e.g., right, right+A, left)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 3. Reset the environment to the starting state
    state = env.reset()
    done = False
    
    print("Starting random agent. Close the window to stop.")
    
    # 4. Run a loop taking random actions
    for step in range(1000):
        if done:
            state = env.reset()
            
        # Sample a random action from our 7 possible actions
        action = env.action_space.sample()
        
        # Take the action in the environment
        state, reward, terminated, truncated, info = env.step(action)
        
        # Check if Mario died or reached the flag
        done = terminated or truncated
        
        # Render the game to the screen
        env.render()

    env.close()

if __name__ == '__main__':
    main()
