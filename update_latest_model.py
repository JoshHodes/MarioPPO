import os
import glob
import shutil

MODELS_DIR = './models_v2/'

import re

def get_latest_model(models_dir):
    files = glob.glob(os.path.join(models_dir, 'mario_ppo_*_steps.zip'))
    if not files:
        return None
        
    def extract_steps(filename):
        match = re.search(r'_(\d+)_steps\.zip$', filename)
        return int(match.group(1)) if match else -1
        
    return max(files, key=extract_steps)

def main():
    latest = get_latest_model(MODELS_DIR)
    if not latest:
        print("No models found.")
        return
    
    target = os.path.join(MODELS_DIR, 'mario_ppo_latest.zip')
    print(f"Latest model found: {latest}")
    print(f"Copying to {target}...")
    shutil.copy2(latest, target)
    print("Done! You can now safely run your git commands:")
    print("  git add models_v2/mario_ppo_latest.zip")
    print("  git add models_v2/curriculum_phase.txt")
    print("  git commit -m \"Update latest model\"")
    print("  git push")

if __name__ == "__main__":
    main()
