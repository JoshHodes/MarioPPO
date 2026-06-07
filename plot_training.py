"""
Generate training curve plots from TensorBoard logs and save to docs/.
Produces: docs/training_curves.png
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT_DIR  = './docs'
OUT_FILE = os.path.join(OUT_DIR, 'training_curves.png')

def find_latest_log_dir(base='./logs'):
    dirs = [d for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f'No log directories found under {base}')
    return max(dirs, key=os.path.getmtime)

LOG_DIR = find_latest_log_dir()

SMOOTH_WINDOW = 50    # rolling-average window (raw signal shown faded behind)

# Phase band colours — cycles if more phases than entries
BAND_COLORS = [
    '#4C9BE8', '#5DBE6E', '#E8A23C', '#9B59B6',
    '#E05C5C', '#1ABC9C', '#F39C12', '#E74C3C',
    '#3498DB', '#2ECC71', '#E67E22', '#8E44AD',
]


def load_segment0(acc, tag):
    """Return (steps, values) for the primary training segment (largest monotonic block)."""
    events = acc.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    vals   = np.array([e.value for e in events])
    diffs  = np.diff(steps)
    resets = np.where(diffs < -1_000_000)[0]
    end    = resets[0] + 1 if len(resets) else len(steps)
    return steps[:end], vals[:end]


def smooth(vals, window):
    if window <= 1:
        return vals
    kernel = np.ones(window) / window
    pad    = np.pad(vals, (window - 1, 0), mode='edge')
    return np.convolve(pad, kernel, mode='valid')


def detect_phase_transitions(steps, targets):
    """Return list of (step, target_value) when the flag target changes."""
    transitions = [(steps[0], targets[0])]
    for i in range(1, len(targets)):
        if abs(targets[i] - targets[i-1]) > 0.001:
            transitions.append((steps[i], targets[i]))
    return transitions


def millions(x, _pos):
    return f'{x / 1e6:.0f}M'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Loading TensorBoard logs from {LOG_DIR} ...')
    acc = EventAccumulator(LOG_DIR)
    acc.Reload()

    steps_rew,    rew    = load_segment0(acc, 'rollout/ep_rew_mean')
    steps_flag,   flag   = load_segment0(acc, 'curriculum/flag_rate')
    steps_target, target = load_segment0(acc, 'curriculum/flag_rate_target')

    transitions = detect_phase_transitions(steps_target, target)
    print(f'Detected {len(transitions)} curriculum phase(s) in logs:')
    for i, (s, t) in enumerate(transitions):
        print(f'  Phase {i}: step {s/1e6:.1f}M  target={t:.0%}')

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True,
        gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.06}
    )
    BG   = '#0D1117'
    PANEL= '#161B22'
    GRID = '#21262D'
    TEXT = '#C9D1D9'
    DIM  = '#8B949E'

    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')
        ax.yaxis.label.set_color(DIM)
        ax.xaxis.label.set_color(DIM)

    # ── Phase bands + dividers ─────────────────────────────────────────────────
    x_max = steps_rew[-1]
    phase_ends = [t[0] for t in transitions[1:]] + [x_max]

    for i, ((s, t), end) in enumerate(zip(transitions, phase_ends)):
        c = BAND_COLORS[i % len(BAND_COLORS)]
        for ax in (ax1, ax2):
            ax.axvspan(s, end, alpha=0.07, color=c, linewidth=0)
        if i > 0:
            for ax in (ax1, ax2):
                ax.axvline(s, color='#30363D', linewidth=0.9, linestyle='--', alpha=0.6)
        # Label on top panel (axes-coordinate y so it doesn't drift with data)
        mid = (s + end) / 2
        label = f'Ph {i}\n{t:.0%}'
        ax1.text(mid, 0.97, label,
                 transform=ax1.get_xaxis_transform(),
                 ha='center', va='top', fontsize=7.5,
                 color=c, alpha=0.9, linespacing=1.3)

    # ── Panel 1: Mean episode reward ───────────────────────────────────────────
    rew_smooth = smooth(rew, SMOOTH_WINDOW)
    ax1.plot(steps_rew, rew, color='#4C9BE8', alpha=0.15, linewidth=0.5)
    ax1.plot(steps_rew, rew_smooth, color='#4C9BE8', linewidth=2.0,
             label='Mean episode reward (smoothed)')
    ax1.set_ylabel('Mean Episode Reward', fontsize=10)
    ax1.grid(axis='y', color=GRID, linewidth=0.6)
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.3,
               facecolor=PANEL, edgecolor='#30363D', labelcolor=TEXT)

    # ── Panel 2: Flag rate vs target ──────────────────────────────────────────
    flag_smooth = smooth(flag, SMOOTH_WINDOW)
    ax2.plot(steps_flag, flag, color='#5DBE6E', alpha=0.15, linewidth=0.5)
    ax2.plot(steps_flag, flag_smooth, color='#5DBE6E', linewidth=2.0,
             label='Flag completion rate (smoothed)')
    ax2.step(steps_target, target, color='#E05C5C', linewidth=1.6,
             linestyle=':', where='post', label='Phase target', alpha=0.9)
    ax2.set_ylim(-0.02, 1.05)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylabel('Flag Rate', fontsize=10)
    ax2.set_xlabel('Training Steps', fontsize=10, labelpad=6)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
    ax2.grid(axis='y', color=GRID, linewidth=0.6)
    ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.3,
               facecolor=PANEL, edgecolor='#30363D', labelcolor=TEXT)

    # ── Title & footer ─────────────────────────────────────────────────────────
    fig.suptitle('Super Mario Bros — PPO Training Curves', color=TEXT,
                 fontsize=14, fontweight='bold', y=0.995)
    fig.text(0.99, 0.005,
             f'{x_max / 1e6:.0f}M training steps  ·  {len(transitions)} curriculum phases reached',
             ha='right', va='bottom', fontsize=8, color='#484F58')

    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'\nSaved → {OUT_FILE}')


if __name__ == '__main__':
    main()
