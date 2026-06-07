"""
record_demos.py — record human demonstrations of a Mario level for behavioral
cloning. You play; it logs (stacked-observation, action) pairs at the exact
cadence the agent acts (frameskip-4 = 15 decisions/sec) so the data is
in-distribution for the BC net. Only runs that REACH THE FLAG are saved.

Controls:
    ←/→     move          Z / Space   jump (A)        X   run / fire (B)
    R       abort current attempt (discard, restart level)
    Esc / window-close    save collected demos and quit

The display runs at ~60fps for smooth play, but your held buttons are sampled
once per agent-step and held for 4 frames — exactly how the policy will act.

Usage:
    python record_demos.py --level 1-3
    python record_demos.py --level 1-3 --scale 3
Demos are merged into demos/<level>.npz  (keys: obs uint8 [N,84,84,4], actions uint8 [N]).
"""
import os, argparse
import numpy as np
import pygame

from bc_common import (make_level_env, reset_env, step_env, process_frame,
                       FrameStacker, keys_to_action, SKIP, SIMPLE_MOVEMENT)


def load_existing(path):
    if os.path.exists(path):
        d = np.load(path)
        return list(d['obs']), list(d['actions'])
    return [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='1-3')
    ap.add_argument('--scale', type=int, default=3, help='display upscale factor')
    ap.add_argument('--fps', type=int, default=60, help='display frames/sec')
    ap.add_argument('--out-dir', default='demos')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.level}.npz')
    saved_obs, saved_act = load_existing(out_path)
    start_count = len(saved_obs)
    print(f"Level {args.level} | existing saved samples: {start_count} | output: {out_path}")

    env = make_level_env(args.level)
    W, H = 256, 240
    pygame.init()
    pygame.display.set_caption(f"Record demos — Mario {args.level}")
    screen = pygame.display.set_mode((W * args.scale, H * args.scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 16)
    # Float + center the window under Hyprland (silently no-ops elsewhere).
    os.system(f"hyprctl dispatch setfloating pid:{os.getpid()} >/dev/null 2>&1")
    os.system(f"hyprctl dispatch centerwindow pid:{os.getpid()} >/dev/null 2>&1")

    def blit(rgb):
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if args.scale != 1:
            surf = pygame.transform.scale(surf, (W * args.scale, H * args.scale))
        screen.blit(surf, (0, 0))

    def overlay(lines):
        for i, txt in enumerate(lines):
            screen.blit(font.render(txt, True, (255, 255, 0)), (6, 6 + i * 18))
        pygame.display.flip()

    clears, attempts = 0, 0
    running = True
    while running:
        # ── new attempt ──
        attempts += 1
        stacker = FrameStacker()
        rgb = reset_env(env)
        stacker.reset(process_frame(rgb))
        ep_obs, ep_act = [], []
        done, info, max_x = False, {}, 0
        attempt_live = True

        while attempt_live and running:
            # sample held keys once per agent-step
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    attempt_live = False          # abort, discard this attempt
            if not running or not attempt_live:
                break

            k = pygame.key.get_pressed()
            action = keys_to_action(
                left=k[pygame.K_LEFT], right=k[pygame.K_RIGHT],
                jump=k[pygame.K_z] or k[pygame.K_SPACE], run=k[pygame.K_x])

            # record (obs the agent would act on, action) BEFORE stepping
            ep_obs.append(stacker.get().astype(np.uint8))
            ep_act.append(action)

            # hold the action SKIP frames, render each sub-frame for smoothness
            for _ in range(SKIP):
                rgb, _, done, info = step_env(env, action)
                blit(rgb)
                overlay([f"{args.level}  attempt {attempts}  clears {clears}",
                         f"saved {start_count + len(saved_obs)}  this-run {len(ep_obs)}",
                         f"x={info.get('x_pos',0)}  status={info.get('status','?')}"])
                clock.tick(args.fps)
                if done:
                    break
            stacker.push(process_frame(rgb))
            max_x = max(max_x, int(info.get('x_pos', 0)))

            if done:
                if info.get('flag_get', False):
                    clears += 1
                    saved_obs.extend(ep_obs)
                    saved_act.extend(ep_act)
                    print(f"  CLEAR! attempt {attempts} -> +{len(ep_obs)} samples "
                          f"(total saved {len(saved_obs)}, clears {clears})")
                else:
                    print(f"  died at x={max_x} (attempt {attempts}, discarded)")
                attempt_live = False

    env.close()
    pygame.quit()

    if len(saved_obs) > start_count:
        np.savez_compressed(out_path,
                            obs=np.asarray(saved_obs, dtype=np.uint8),
                            actions=np.asarray(saved_act, dtype=np.uint8))
        print(f"\nSaved {len(saved_obs)} samples ({clears} clears this session) -> {out_path}")
    else:
        print("\nNo new successful runs recorded; nothing saved.")


if __name__ == '__main__':
    main()
