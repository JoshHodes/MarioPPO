"""
harvest_demos.py — auto-generate demonstrations by recording the EXISTING PPO
model's winning runs on a level (no human needed). For levels the old PPO already
clears (1-1, 1-2, 1-4, 2-3, 2-4), its successful episodes are perfectly good
demonstrations: same obs pipeline, same action space as the BC net. Only
flag-reaching episodes are kept.

Output is the SAME format as record_demos.py (demos/<level>.npz: obs uint8
[N,84,84,4], actions uint8 [N]) so behavioral_clone.py / finetune_bc.py consume
it unchanged.

Usage:
    python harvest_demos.py --level 1-1 --episodes 20
    python harvest_demos.py --level 2-3 --episodes 20 --max-attempts 200
"""
import os, argparse
import numpy as np
from stable_baselines3 import PPO
from train import make_venv, get_latest_model, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', required=True)
    ap.add_argument('--model', default=None, help='defaults to latest in ./models_v2')
    ap.add_argument('--episodes', type=int, default=20, help='target SUCCESSFUL (flag) episodes')
    ap.add_argument('--max-attempts', type=int, default=300)
    ap.add_argument('--out-dir', default='demos')
    ap.add_argument('--append', action='store_true', help='merge into existing demos/<level>.npz')
    args = ap.parse_args()

    model_path = args.model or get_latest_model('./models_v2')
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.level}.npz')

    saved_obs, saved_act = [], []
    if args.append and os.path.exists(out_path):
        d = np.load(out_path); saved_obs, saved_act = list(d['obs']), list(d['actions'])

    venv = make_venv('SuperMarioBrosRandomStages-v0', [args.level], 1)
    model = PPO.load(model_path, device=DEVICE)
    print(f"Harvesting {args.level} from {os.path.basename(model_path)} | "
          f"target {args.episodes} clears (max {args.max_attempts} attempts) | device {DEVICE}")

    obs = venv.reset()
    ep_obs, ep_act = [], []
    successes, attempts = 0, 0
    while successes < args.episodes and attempts < args.max_attempts:
        action, _ = model.predict(obs, deterministic=False)   # PPO relies on entropy
        ep_obs.append(obs[0].astype(np.uint8))                # (84,84,4) the model acted on
        ep_act.append(int(action[0]))
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            attempts += 1
            if infos[0].get('flag_get', False):
                successes += 1
                saved_obs.extend(ep_obs); saved_act.extend(ep_act)
                print(f"  clear {successes:3d}/{args.episodes}  (+{len(ep_obs)} samples, "
                      f"attempt {attempts})", flush=True)
            ep_obs, ep_act = [], []
    venv.close()

    if successes == 0:
        print(f"FAILED: 0 clears in {attempts} attempts — PPO can't clear {args.level}; skip it.")
        return 1
    np.savez_compressed(out_path,
                        obs=np.asarray(saved_obs, dtype=np.uint8),
                        actions=np.asarray(saved_act, dtype=np.uint8))
    print(f"Saved {len(saved_obs)} samples from {successes} clears "
          f"(clear rate {successes/max(attempts,1):.0%}) -> {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
