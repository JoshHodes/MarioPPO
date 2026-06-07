"""
behavioral_clone.py — train a reactive CNN to imitate human demonstrations
(behavioral cloning), then evaluate whether it clears the level.

This is the proof-of-concept for the imitation route: can supervised learning
on a handful of human clears crack a level (1-3) that PPO + RND curiosity could
never move off 0%? The trained net is a genuine learned policy — at eval it sees
only stacked grayscale frames and outputs a button press, no emulator look-ahead.

Usage:
    python behavioral_clone.py --level 1-3                 # train + eval
    python behavioral_clone.py --level 1-3 --epochs 40 --eval-episodes 30
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn

from bc_common import BCPolicy, NUM_ACTIONS, rollout, SIMPLE_MOVEMENT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_demos(path):
    d = np.load(path)
    obs, act = d['obs'], d['actions']
    print(f"Loaded {len(obs)} samples from {path}")
    counts = np.bincount(act, minlength=NUM_ACTIONS)
    for i, c in enumerate(counts):
        print(f"  action {i} {str(SIMPLE_MOVEMENT[i]):20s}: {c:6d} "
              f"({100*c/max(len(act),1):4.1f}%)")
    return obs, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='1-3')
    ap.add_argument('--demos', default=None, help='defaults to demos/<level>.npz')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--eval-episodes', type=int, default=30)
    ap.add_argument('--out', default=None, help='defaults to models_v2/bc_<level>.pt')
    args = ap.parse_args()

    demos_path = args.demos or f'demos/{args.level}.npz'
    out_path = args.out or f'models_v2/bc_{args.level}.pt'
    obs, act = load_demos(demos_path)

    # tensors: obs (N,84,84,4)uint8 -> (N,4,84,84) float[0,1]
    X = torch.from_numpy(np.transpose(obs.astype(np.float32) / 255.0, (0, 3, 1, 2)))
    y = torch.from_numpy(act.astype(np.int64))
    n = len(X)
    perm = torch.randperm(n)
    X, y = X[perm], y[perm]
    n_val = int(n * args.val_frac)
    Xtr, ytr, Xval, yval = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    # class weights — demos are dominated by "run right"; weight rarer actions up
    counts = torch.bincount(ytr, minlength=NUM_ACTIONS).float()
    weights = (counts.sum() / (counts.clamp(min=1) * NUM_ACTIONS)).to(DEVICE)

    policy = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"\nTraining on {len(Xtr)} samples (val {len(Xval)}) | device={DEVICE}")

    for ep in range(1, args.epochs + 1):
        policy.train()
        idx = torch.randperm(len(Xtr))
        tot, correct, lsum = 0, 0, 0.0
        for i in range(0, len(Xtr), args.batch):
            b = idx[i:i + args.batch]
            xb, yb = Xtr[b].to(DEVICE), ytr[b].to(DEVICE)
            logits = policy(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item() * len(b)
            correct += (logits.argmax(1) == yb).sum().item(); tot += len(b)
        # val acc
        policy.eval()
        with torch.no_grad():
            vcorrect, vtot = 0, 0
            for i in range(0, len(Xval), args.batch):
                xb, yb = Xval[i:i+args.batch].to(DEVICE), yval[i:i+args.batch].to(DEVICE)
                vcorrect += (policy(xb).argmax(1) == yb).sum().item(); vtot += len(yb)
        vacc = vcorrect / max(vtot, 1)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss {lsum/tot:.3f}  train_acc {correct/tot:.1%}  val_acc {vacc:.1%}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'state_dict': policy.state_dict(), 'level': args.level}, out_path)
    print(f"\nSaved BC policy -> {out_path}")

    # ── the real test: does it clear the level? ──
    print(f"\nEvaluating on {args.level} ({args.eval_episodes} episodes each)...")
    fr_d, mx_d, _ = rollout(policy, args.level, args.eval_episodes, DEVICE, deterministic=True)
    fr_s, mx_s, _ = rollout(policy, args.level, args.eval_episodes, DEVICE, deterministic=False)
    print(f"  deterministic : flag_rate {fr_d:5.1%}  mean_max_x {mx_d:6.0f}")
    print(f"  stochastic    : flag_rate {fr_s:5.1%}  mean_max_x {mx_s:6.0f}")
    print("\nReference: PPO + RND never cleared 1-3 (0%, stalled at x~765).")


if __name__ == '__main__':
    main()
