"""
brute_force_campaign.py
Sample many STOCHASTIC full-game runs (1-1 -> 8-4, 3 shared lives) in parallel
and save a video of the first run that beats 8-4 -- or, if none clears, the
furthest run reached.

Rationale: a *reliable* single-run clear is extremely hard, but even a tiny
per-run success probability is enough if you sample enough diverse runs and
record the winner. Runs must be STOCHASTIC -- a deterministic (greedy) policy
replays the identical run every time, so parallelism would gain nothing.
--temperature > 1 flattens the policy to widen the spread of attempts.

Usage:
    python brute_force_campaign.py                       # 24 envs, up to 5000 runs, stop at first clear
    python brute_force_campaign.py --envs 24 --attempts 20000
    python brute_force_campaign.py --temperature 1.3     # more diverse attempts
    python brute_force_campaign.py --model models_v2/mario_ppo_83959584_steps.zip
    python brute_force_campaign.py --no-stop-on-clear    # run all attempts, keep best

Outputs (in ./clips/):
    campaign_clear_<ts>.mp4 / .npy   if 8-4 is beaten
    campaign_best_<ts>.mp4  / .npy   the furthest run otherwise
The .npy holds the agent-level action sequence so any run can be re-rendered later.
"""
import os
import argparse
import time
import numpy as np
import torch
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from train import make_campaign_venv, get_latest_model, DEVICE


def sample_actions(model, obs, temperature):
    """Stochastic action per env. temperature>1 flattens the policy for diversity."""
    if abs(temperature - 1.0) < 1e-6:
        actions, _ = model.predict(obs, deterministic=False)
        return actions
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits           # (n_envs, n_actions)
        probs = torch.softmax(logits / temperature, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return actions.cpu().numpy()


def render_actions_to_video(actions, out_path, fps=60):
    """Replay an agent-level action sequence on a fresh full-game env and stream
    every game frame to an mp4. NES SMB is deterministic given the inputs, so the
    replay reproduces the sampled run. Each agent action is held for 4 frames to
    match the training frame-skip. Returns True if the replay reached the 8-4 flag.
    """
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r

    h, w = obs.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    cleared = False
    for a in actions:
        broke = False
        for _ in range(4):                          # replay frame-skip
            out = env.step(int(a))
            obs, info = out[0], out[-1]
            term = out[2]
            trunc = out[3] if len(out) == 5 else False
            writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            if info.get('flag_get') and info.get('world') == 8 and info.get('stage') == 4:
                cleared = True
            if term or trunc:
                broke = True
                break
        if broke:
            break
    writer.release()
    env.close()
    return cleared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', type=int, default=24, help='parallel full-game envs')
    ap.add_argument('--attempts', type=int, default=5000, help='max full-game runs to sample')
    ap.add_argument('--temperature', type=float, default=1.0, help='>1 = more diverse attempts')
    ap.add_argument('--model', type=str, default=None, help='default: latest checkpoint')
    ap.add_argument('--no-stop-on-clear', dest='stop_on_clear', action='store_false',
                    help='keep sampling all attempts instead of quitting at the first clear')
    ap.add_argument('--out', type=str, default='./clips')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_path = args.model or get_latest_model('./models_v2')
    if not model_path:
        print('No model found in ./models_v2'); return
    print(f'Model       : {os.path.basename(model_path)}')
    print(f'Envs        : {args.envs}   Attempts cap: {args.attempts}   Temp: {args.temperature}')
    print(f'Stop on clear: {args.stop_on_clear}\n')

    model = PPO.load(model_path, device=DEVICE)
    venv = make_campaign_venv(args.envs)
    obs = venv.reset()

    action_bufs = [[] for _ in range(args.envs)]
    far_ws      = [(1, 1)] * args.envs              # running furthest (world,stage) per env
    completed = 0
    clears = 0
    last_print = 0
    best = {'key': (1, 1), 'stages': -1, 'actions': None}
    winner = None
    t0 = time.time()

    while completed < args.attempts:
        actions = sample_actions(model, obs, args.temperature)
        for i, a in enumerate(actions):
            action_bufs[i].append(int(a))
        obs, _, dones, infos = venv.step(actions)

        for i in range(args.envs):
            info = infos[i]
            ws = (info.get('world', 1), info.get('stage', 1))
            if ws > far_ws[i]:
                far_ws[i] = ws
            if dones[i]:
                completed += 1
                cleared = bool(info.get('campaign_cleared', False))
                stages  = int(info.get('stages_cleared', 0))
                seq = list(action_bufs[i])
                if cleared and winner is None:
                    clears += 1
                    winner = seq
                elif cleared:
                    clears += 1
                if (far_ws[i], stages) > (best['key'], best['stages']):
                    best = {'key': far_ws[i], 'stages': stages, 'actions': seq}
                action_bufs[i] = []
                far_ws[i] = (1, 1)

        if completed - last_print >= args.envs:
            last_print = completed
            el = max(1e-6, time.time() - t0)
            fw = best['key']
            print(f'  attempts={completed:6d}  clears={clears}  best={fw[0]}-{fw[1]} '
                  f'({best["stages"]} stages)  {completed/el:.1f} runs/s', flush=True)

        if winner is not None and args.stop_on_clear:
            print(f'\n*** 8-4 CLEARED around attempt {completed}! ***')
            break

    venv.close()
    ts = time.strftime('%Y%m%d_%H%M%S')

    if winner is not None:
        seq, tag = winner, 'clear'
    else:
        seq, tag = best['actions'], 'best'
        fw = best['key']
        print(f'\nNo full clear in {completed} attempts. Furthest reached: '
              f'{fw[0]}-{fw[1]} ({best["stages"]} stages). Saving that run.')

    if not seq:
        print('No completed runs to save.'); return

    npy_path = os.path.join(args.out, f'campaign_{tag}_{ts}.npy')
    mp4_path = os.path.join(args.out, f'campaign_{tag}_{ts}.mp4')
    np.save(npy_path, np.array(seq, dtype=np.int16))
    print(f'Rendering {len(seq)} actions -> {mp4_path} ...')
    ok = render_actions_to_video(seq, mp4_path)
    if tag == 'clear':
        note = 'reproduced the 8-4 clear' if ok else 'WARNING: replay did not re-clear (nondeterminism?)'
    else:
        note = 'complete'
    print(f'Saved video  : {mp4_path}  ({note})')
    print(f'Saved actions: {npy_path}')
    print(f'\nTotal: {completed} attempts, {clears} full clears, '
          f'{completed/max(1e-6, time.time()-t0):.1f} runs/s')


if __name__ == '__main__':
    main()
