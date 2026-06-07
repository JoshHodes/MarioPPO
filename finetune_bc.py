"""
finetune_bc.py — RL-fine-tune a behavioral-cloned policy with PPO (option 1).

Pure BC got 1-3 past the PPO+RND wall (mean_max_x ~972 vs ~765) but never
reached the flag: it plays the first ~2/3 of the level correctly, then dies in
the later third on states the demos didn't cover (distribution shift). PPO from
scratch couldn't DISCOVER 1-3 at all — but starting from the BC policy it no
longer has to: it already does the right thing for most of the level, so PPO
only needs to REFINE the last stretch, which is local exploration it can do.

Pipeline:
  1. build an SB3 PPO with CnnPolicy (net_arch=[] so it matches BCPolicy 1:1),
  2. transfer the BC weights into the features extractor + action head,
  3. VERIFY the transfer (SB3 must reproduce BC's actions on the demo frames),
  4. fine-tune on the single level with the dense shaped reward,
  5. log rolling flag-rate / max-x, save the best-by-flag-rate checkpoint.

Usage:
    python finetune_bc.py --level 1-3
    python finetune_bc.py --level 1-3 --steps 3000000 --n-envs 8
"""
import os
# Pin BLAS/OpenMP to 1 thread BEFORE numpy/torch import. The workload is
# CPU-env-bound across many SubprocVecEnv worker processes; letting each
# worker's numpy/BLAS spin up its own thread pool oversubscribes the 16 threads
# once n_envs is pushed up. The GPU (idle at ~7%) does the policy compute, so
# the main process doesn't need CPU math threads either.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
from collections import deque
import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from train import make_venv, DEVICE
from bc_common import BCPolicy


def transfer_bc_into_ppo(model, bc_state):
    """Copy BC conv trunk + 512-head into the SB3 policy's three feature-extractor
    copies and action_net. value_net is left random (BC has no value head)."""
    sd = model.policy.state_dict()
    for prefix in ('features_extractor', 'pi_features_extractor', 'vf_features_extractor'):
        if f'{prefix}.cnn.0.weight' not in sd:
            continue
        for layer in ('cnn.0', 'cnn.2', 'cnn.4'):
            sd[f'{prefix}.{layer}.weight'] = bc_state[f'{layer}.weight'].clone()
            sd[f'{prefix}.{layer}.bias']   = bc_state[f'{layer}.bias'].clone()
        sd[f'{prefix}.linear.0.weight'] = bc_state['head.0.weight'].clone()
        sd[f'{prefix}.linear.0.bias']   = bc_state['head.0.bias'].clone()
    sd['action_net.weight'] = bc_state['head.2.weight'].clone()
    sd['action_net.bias']   = bc_state['head.2.bias'].clone()
    model.policy.load_state_dict(sd)


def verify_transfer(model, bc_state, demos_path, device, n=512):
    """SB3 policy must reproduce BC's argmax actions on the demo frames. If they
    don't agree, the frame-order / normalization is mismatched — abort early."""
    d = np.load(demos_path)
    obs = d['obs'][:n]                                   # (n,84,84,4) uint8
    bc = BCPolicy().to(device); bc.load_state_dict(bc_state); bc.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.transpose(obs.astype(np.float32) / 255.0,
                                          (0, 3, 1, 2))).to(device)
        bc_act = bc(x).argmax(1).cpu()
        obs_t, _ = model.policy.obs_to_tensor(obs)       # SB3 transpose+to-tensor
        dist = model.policy.get_distribution(obs_t)
        sb3_act = dist.distribution.probs.argmax(1).cpu()
    agree = (bc_act == sb3_act).float().mean().item()
    return agree


class StabilizeCallback(BaseCallback):
    """Two stabilization mechanisms (the fix for spike-then-collapse):
      1. ANNEAL exploration: ent_coef held high to break through, then decayed
         so the policy consolidates onto cleared behavior instead of bouncing.
      2. REAL-EVAL checkpointing: every `eval_every` steps, run a proper N-episode
         eval on a held-out env and save the best by THAT — not the noisy training
         window (whose 'best' was a selection-biased fluke that clean-eval'd to 0%).
    """
    def __init__(self, save_path, level, total_steps, ent_start, ent_end,
                 hold_frac=0.5, eval_every=250_000, eval_episodes=15, window=100,
                 log_every=25000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.level = level
        self.total = total_steps
        self.ent_start, self.ent_end = ent_start, ent_end
        self.hold_frac = hold_frac
        self.eval_every, self.eval_episodes = eval_every, eval_episodes
        self.flags = deque(maxlen=window)
        self.maxx = deque(maxlen=window)
        self.log_every = log_every
        self._next_log = log_every
        self._next_eval = eval_every
        self.best_real = -1.0
        self.eval_venv = None

    def _on_training_start(self):
        self.eval_venv = make_venv('SuperMarioBrosRandomStages-v0', [self.level], 1)

    def _ent_now(self):
        frac = min(self.num_timesteps / self.total, 1.0)   # hold high then decay
        if frac < self.hold_frac:
            return self.ent_start
        t = (frac - self.hold_frac) / max(1e-6, 1.0 - self.hold_frac)
        return self.ent_start + t * (self.ent_end - self.ent_start)

    def _real_eval(self):
        flags = []
        obs = self.eval_venv.reset()
        while len(flags) < self.eval_episodes:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, _, dones, infos = self.eval_venv.step(action)
            if dones[0]:
                flags.append(1 if infos[0].get('flag_get', False) else 0)
        return float(np.mean(flags))

    def _on_step(self):
        # anneal exploration
        self.model.ent_coef = self._ent_now()
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done:
                self.flags.append(1 if info.get('flag_get', False) else 0)
                self.maxx.append(int(info.get('x_pos', 0)))
        if self.num_timesteps >= self._next_log:
            self._next_log += self.log_every
            fr = float(np.mean(self.flags)) if self.flags else 0.0
            mx = float(np.mean(self.maxx)) if self.maxx else 0.0
            print(f"[{self.num_timesteps:>9}] train_flag={fr:5.1%}  avg_max_x={mx:6.0f}  "
                  f"ent={self.model.ent_coef:.4f}  best_real={self.best_real:5.1%}", flush=True)
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            real = self._real_eval()
            tag = ""
            if real > self.best_real:
                self.best_real = real
                self.model.save(self.save_path)
                tag = f" -> NEW BEST, saved {self.save_path}"
            print(f"  [eval @ {self.num_timesteps}] real_flag_rate={real:.1%} "
                  f"(n={self.eval_episodes}){tag}", flush=True)
        return True

    def _on_training_end(self):
        if self.eval_venv is not None:
            self.eval_venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='1-3')
    ap.add_argument('--bc', default=None, help='defaults to models_v2/bc_<level>.pt')
    ap.add_argument('--steps', type=int, default=4_000_000,
                    help='more room: hold exploration high to break through, then anneal to consolidate')
    ap.add_argument('--n-envs', type=int, default=24,
                    help='24 = proven local sweet spot (~1400 fps ceiling on 8c/16t); '
                         'workers block on the sync barrier so >threads fills the gaps')
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--ent-start', type=float, default=0.02, help='high to break the wall')
    ap.add_argument('--ent-end', type=float, default=0.004, help='low to consolidate/stabilize')
    ap.add_argument('--hold-frac', type=float, default=0.5,
                    help='fraction of training to hold ent_start before decaying to ent_end '
                         '(lower = more consolidation time at low entropy)')
    ap.add_argument('--clip-start', type=float, default=0.2)
    ap.add_argument('--clip-end', type=float, default=0.12)
    args = ap.parse_args()

    bc_path = args.bc or f'models_v2/bc_{args.level}.pt'
    demos_path = f'demos/{args.level}.npz'
    best_path = f'models_v2/bc_ft_{args.level}_best'
    final_path = f'models_v2/bc_ft_{args.level}_final'
    bc_state = torch.load(bc_path, map_location=DEVICE)['state_dict']
    print(f"BC init: {bc_path} | level {args.level} | n_envs {args.n_envs} | device {DEVICE}")

    # clip_range as an SB3 schedule (progress_remaining: 1->0), linear start->end
    clip_sched = lambda pr: args.clip_end + pr * (args.clip_start - args.clip_end)

    venv = make_venv('SuperMarioBrosRandomStages-v0', [args.level], args.n_envs)
    model = PPO('CnnPolicy', venv, policy_kwargs=dict(net_arch=[]), device=DEVICE,
                learning_rate=args.lr, ent_coef=args.ent_start, clip_range=clip_sched,
                n_steps=512, batch_size=256, n_epochs=6, gamma=0.99, gae_lambda=0.95,
                verbose=0)

    transfer_bc_into_ppo(model, bc_state)
    agree = verify_transfer(model, bc_state, demos_path, DEVICE)
    print(f"Transfer check: SB3 reproduces BC actions on {agree:.1%} of demo frames "
          f"({'OK' if agree > 0.98 else 'MISMATCH — aborting'})")
    if agree <= 0.98:
        venv.close()
        return

    print(f"\nFine-tuning {args.steps:,} steps | ent {args.ent_start}->{args.ent_end} (hold then anneal) "
          f"| clip {args.clip_start}->{args.clip_end} | real-eval checkpointing.\n"
          f"Goal: a STABLE clearing policy (prior run spiked to 38% then collapsed to 0% on clean eval).\n")
    cb = StabilizeCallback(save_path=best_path, level=args.level, total_steps=args.steps,
                           ent_start=args.ent_start, ent_end=args.ent_end, hold_frac=args.hold_frac)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(final_path)
    print(f"\nDone. best REAL flag_rate {cb.best_real:.1%}. Saved {best_path} (best by real eval) + {final_path}.")


if __name__ == '__main__':
    main()
