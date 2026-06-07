"""
eval_ft.py — clean evaluation of a fine-tuned SB3 PPO checkpoint on one level.
Gives the HONEST flag-rate over N independent episodes (vs the selection-biased
"best window" the training callback reports).

Usage:
    python eval_ft.py --model models_v2/bc_ft_1-3_best.zip --level 1-3 --episodes 50
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from train import make_venv, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--level', default='1-3')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--deterministic', action='store_true',
                    help='default is stochastic (policy relies on exploration entropy)')
    args = ap.parse_args()

    venv = make_venv('SuperMarioBrosRandomStages-v0', [args.level], 1)
    model = PPO.load(args.model, device=DEVICE)
    print(f"Eval {args.model} on {args.level} | {args.episodes} eps | "
          f"{'deterministic' if args.deterministic else 'stochastic'}")

    flags, maxxs = [], []
    obs = venv.reset()
    cur_max = 0
    while len(flags) < args.episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, dones, infos = venv.step(action)
        cur_max = max(cur_max, int(infos[0].get('x_pos', 0)))
        if dones[0]:
            flags.append(1 if infos[0].get('flag_get', False) else 0)
            maxxs.append(cur_max)
            cur_max = 0
            done_n = len(flags)
            if done_n % 10 == 0:
                print(f"  {done_n:3d} eps: flag_rate so far {np.mean(flags):.1%}")
    venv.close()

    fr = np.mean(flags)
    print(f"\n=== {args.level}: flag_rate {fr:.1%} ({sum(flags)}/{len(flags)})  "
          f"mean_max_x {np.mean(maxxs):.0f}  (95% CI ±{1.96*np.sqrt(fr*(1-fr)/len(flags)):.1%}) ===")


if __name__ == '__main__':
    main()
