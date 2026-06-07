"""
explore_rnd.py — controlled experiment: can intrinsic-curiosity exploration (RND,
Random Network Distillation) crack a hard-exploration level that plain PPO is
stuck at 0% on (e.g. 1-3's precise jumps over pits)?

RND adds a novelty bonus = prediction error of a trained network against a fixed
random target network. Reaching rarely-seen states (e.g. being airborne over the
pit, landing on a new platform) yields reward even before the agent reliably
clears, which can bootstrap learning a precise sequence pure PPO won't stumble on.

This is a THROWAWAY diagnostic: it trains on ONE level (forgets others) purely to
test whether RND moves that level off 0%. If it works, RND gets integrated into
the full curriculum run; if not, that's strong evidence the wall is beyond this
method.

Usage:
    python explore_rnd.py --level 1-3 --steps 4000000
    python explore_rnd.py --level 2-1 --int-coef 1.0 --ent-coef 0.05
"""
import os, argparse, glob
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback

from train import make_venv, get_latest_model, DEVICE


# ─── RND networks ───────────────────────────────────────────────────────────
class RNDNet(nn.Module):
    """Nature-CNN trunk -> embedding. Target is frozen-random; predictor learns."""
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x):
        bmean, bvar, bcount = x.mean(0), x.var(0), x.shape[0]
        d = bmean - self.mean
        tot = self.count + bcount
        self.mean += d * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        M2 = m_a + m_b + d**2 * self.count * bcount / tot
        self.var = M2 / tot
        self.count = tot


class RNDRewardWrapper(VecEnvWrapper):
    """Adds normalized intrinsic reward to extrinsic; trains the predictor online.
    PPO sees only the summed reward, so no advantage-recomputation surgery."""
    def __init__(self, venv, int_coef=0.5, lr=1e-4, device='cpu'):
        super().__init__(venv)
        self.int_coef = int_coef
        self.device = device
        self.target = RNDNet().to(device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = RNDNet().to(device)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd((4, 84, 84))
        self.int_rms = RunningMeanStd(())
        self._log_int = deque(maxlen=2000)

    def _prep(self, obs):
        # (N,84,84,4) uint8  ->  (N,4,84,84) float
        x = np.transpose(obs.astype(np.float32), (0, 3, 1, 2))
        return x

    def _normalize(self, x):
        m = self.obs_rms.mean.astype(np.float32)
        s = np.sqrt(self.obs_rms.var).astype(np.float32) + 1e-8
        return np.clip((x - m) / s, -5, 5)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        x = self._prep(obs)
        self.obs_rms.update(x)
        xn = torch.as_tensor(self._normalize(x), device=self.device)

        # intrinsic = per-env MSE(predictor, target) in embedding space
        with torch.no_grad():
            err = (self.predictor(xn) - self.target(xn)).pow(2).mean(dim=1)
        intr = err.cpu().numpy()
        self.int_rms.update(intr)
        intr_norm = intr / (np.sqrt(self.int_rms.var) + 1e-8)
        self._log_int.extend(intr_norm.tolist())

        # train predictor toward target on this batch
        pred = self.predictor(xn)
        with torch.no_grad():
            tgt = self.target(xn)
        loss = (pred - tgt).pow(2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        rews = rews + self.int_coef * intr_norm.astype(rews.dtype)
        return obs, rews, dones, infos

    def mean_intrinsic(self):
        return float(np.mean(self._log_int)) if self._log_int else 0.0


class FlagRateCallback(BaseCallback):
    """Logs rolling flag-completion rate + mean intrinsic reward."""
    def __init__(self, rnd_wrapper, window=100, every=20480):
        super().__init__()
        self.rnd = rnd_wrapper
        self.flags = deque(maxlen=window)
        self.maxx = deque(maxlen=window)
        self.every = every
        self._next = None

    def _on_training_start(self):
        self._next = self.num_timesteps + self.every

    def _on_step(self):
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done:
                self.flags.append(1 if info.get('flag_get', False) else 0)
                self.maxx.append(int(info.get('x_pos', 0)))
        if self.num_timesteps >= self._next:
            self._next += self.every
            fr = np.mean(self.flags) if self.flags else 0.0
            mx = np.mean(self.maxx) if self.maxx else 0.0
            print(f"[{self.num_timesteps:>9}] flag_rate={fr:5.1%} (n={len(self.flags):3d})  "
                  f"avg_max_x={mx:6.0f}  mean_intrinsic={self.rnd.mean_intrinsic():.3f}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='1-3')
    ap.add_argument('--steps', type=int, default=4_000_000)
    ap.add_argument('--n-envs', type=int, default=16)
    ap.add_argument('--int-coef', type=float, default=0.5)
    ap.add_argument('--ent-coef', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--resume-from', default=None)
    args = ap.parse_args()

    ckpt = args.resume_from or get_latest_model('./models_v2')
    print(f"Level: {args.level} | seed: {os.path.basename(ckpt)} | int_coef={args.int_coef} "
          f"| ent_coef={args.ent_coef} | device={DEVICE}")

    venv = make_venv('SuperMarioBrosRandomStages-v0', [args.level], args.n_envs)
    rnd = RNDRewardWrapper(venv, int_coef=args.int_coef, device=DEVICE)

    model = PPO.load(ckpt, env=rnd, device=DEVICE, custom_objects={
        'learning_rate': args.lr, 'ent_coef': args.ent_coef,
        'n_steps': 2048, 'batch_size': 512, 'n_epochs': 4, 'clip_range': 0.15,
    })
    cb = FlagRateCallback(rnd)
    print(f"Baseline (this checkpoint) is 0% on {args.level}. Watching for movement...\n")
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=False)
    model.save(f"./models_v2/rnd_experiment_{args.level.replace('-','_')}")
    print("Done. (throwaway single-level model saved as rnd_experiment_*)")


if __name__ == '__main__':
    main()
