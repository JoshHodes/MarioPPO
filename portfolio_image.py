"""
Generate a portfolio image showing the RL preprocessing pipeline:
  Original RGB frame  →  Greyscale 84×84  →  4-frame stack
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # headless rendering

import numpy as np
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── 1. Boot env and collect 4 consecutive frames ──────────────────────────────

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, _ = env.reset()

# Walk forward for a handful of steps so the scene is interesting
raw_frames = []
for i in range(60):
    action = 1  # run right
    obs, _, done, _, _ = env.step(action)
    if i >= 56:          # grab the last 4 frames
        raw_frames.append(obs.copy())
    if done:
        obs, _ = env.reset()

env.close()

# ── 2. Preprocess each frame the same way MarioWrapper does ──────────────────

def preprocess(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (84, 84), interpolation=cv2.INTER_AREA)
    return grey, small          # full-res grey (for display), 84×84

raw_frames_np   = [np.array(f) for f in raw_frames]
grey_full_list  = []
grey_84_list    = []
for f in raw_frames_np:
    gf, g84 = preprocess(f)
    grey_full_list.append(gf)
    grey_84_list.append(g84)

# ── 3. Build the figure ───────────────────────────────────────────────────────

BG      = "#0d0d0d"
ACCENT  = "#e8c84a"        # golden yellow — Mario-ish
LABEL   = "#ffffff"
DIMTEXT = "#888888"
ARROW   = "#555555"

fig = plt.figure(figsize=(16, 5.2), facecolor=BG)

# Column layout (left-pad | rgb | gap | grey84 | gap | 4×stack | right-pad)
# Using gridspec with width ratios
gs = fig.add_gridspec(
    1, 9,
    left=0.03, right=0.97, top=0.80, bottom=0.12,
    width_ratios=[0.05, 1.6, 0.25, 1.0, 0.25, 1.0, 0.08, 1.0, 0.08],
    wspace=0,
)

# --- Panel A: original RGB ---------------------------------------------------
ax_rgb = fig.add_subplot(gs[0, 1])
ax_rgb.imshow(raw_frames_np[-1])
ax_rgb.set_xticks([])
ax_rgb.set_yticks([])
for spine in ax_rgb.spines.values():
    spine.set_edgecolor(ACCENT)
    spine.set_linewidth(2)

# --- Panel B: 84×84 greyscale ------------------------------------------------
ax_g = fig.add_subplot(gs[0, 3])
ax_g.imshow(grey_84_list[-1], cmap="gray", vmin=0, vmax=255)
ax_g.set_xticks([])
ax_g.set_yticks([])
for spine in ax_g.spines.values():
    spine.set_edgecolor(ACCENT)
    spine.set_linewidth(2)

# --- Panel C: 4-frame stack (side by side, separated by thin gaps) -----------
# We'll create a composite image: 4 × 84 wide, 84 tall, with 3-px separators
SEP = 3
composite = np.full((84, 4 * 84 + 3 * SEP), 40, dtype=np.uint8)
for i, frame in enumerate(grey_84_list):
    x = i * (84 + SEP)
    composite[:, x:x + 84] = frame

ax_stack = fig.add_subplot(gs[0, 5:8])
ax_stack.imshow(composite, cmap="gray", vmin=0, vmax=255, aspect="auto")
ax_stack.set_xticks([])
ax_stack.set_yticks([])
for spine in ax_stack.spines.values():
    spine.set_edgecolor(ACCENT)
    spine.set_linewidth(2)

# --- Arrows between panels ---------------------------------------------------
arrow_kw = dict(
    arrowstyle="->",
    color=ARROW,
    lw=2,
    mutation_scale=18,
)

def add_arrow(fig, ax_left, ax_right):
    """Draw an arrow in figure coords from right edge of ax_left to left edge of ax_right."""
    fig.canvas.draw()
    left_bb  = ax_left.get_position()
    right_bb = ax_right.get_position()
    mid_y    = (left_bb.y0 + left_bb.y1) / 2
    x0 = left_bb.x1 + 0.005
    x1 = right_bb.x0 - 0.005
    ax_fig = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_fig.set_xlim(0, 1)
    ax_fig.set_ylim(0, 1)
    ax_fig.axis("off")
    ax_fig.annotate(
        "",
        xy=(x1, mid_y), xytext=(x0, mid_y),
        xycoords="figure fraction", textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=ARROW, lw=2, mutation_scale=18),
    )

add_arrow(fig, ax_rgb, ax_g)
add_arrow(fig, ax_g, ax_stack)

# --- Title / subtitle --------------------------------------------------------
fig.text(0.5, 0.97, "Super Mario Bros — PPO Agent",
         ha="center", va="top", color=LABEL,
         fontsize=17, fontweight="bold", fontfamily="monospace")
fig.text(0.5, 0.905, "observation preprocessing pipeline",
         ha="center", va="top", color=DIMTEXT,
         fontsize=10, fontfamily="monospace")

# --- Labels above each panel -------------------------------------------------
label_y = 0.815     # in figure fraction

def fig_label(fig, ax, text, sub, accent_color=ACCENT):
    pos = ax.get_position()
    cx  = (pos.x0 + pos.x1) / 2
    fig.text(cx, label_y,      text, ha="center", va="bottom",
             color=accent_color, fontsize=13, fontweight="bold",
             fontfamily="monospace")
    fig.text(cx, label_y - 0.065, sub, ha="center", va="bottom",
             color=DIMTEXT, fontsize=9, fontfamily="monospace")

fig.canvas.draw()
fig_label(fig, ax_rgb,   "RGB observation",   "240 × 256 × 3")
fig_label(fig, ax_g,     "greyscale 84×84",   "agent input (1 frame)")
fig_label(fig, ax_stack, "4-frame stack",      "what the network sees  (84 × 84 × 4)")

# ── 4. Save ──────────────────────────────────────────────────────────────────
out = "docs/portfolio.png"
os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
