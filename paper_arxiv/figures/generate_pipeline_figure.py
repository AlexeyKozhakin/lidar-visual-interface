"""
Figure 1 — Pipeline block diagram for paper_draft_v2.md
Run: python paper_arxiv/figures/generate_pipeline_figure.py
Output: paper_arxiv/figures/figure1_pipeline.png  (300 dpi)
         paper_arxiv/figures/figure1_pipeline.svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT   = '#37474F'   # dark slate  – input node
C_PIPE    = '#1565C0'   # deep blue   – main pipeline stages
C_FORK    = '#455A64'   # blue-grey   – stitching / fork node
C_POLY    = '#2E7D32'   # forest green – polygon branch
C_CLASS   = '#6A1B9A'   # deep purple  – 3D classification branch
C_OUT_P   = '#1B5E20'   # dark green   – shapefile output
C_OUT_C   = '#4A148C'   # dark purple  – classified LAS output
C_BG      = '#F8F9FA'   # near-white background
C_TEXT    = 'white'
C_ARROW   = '#546E7A'

# ── helper functions ──────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, title, subtitle='', color=C_PIPE, fs=8.5, fs2=7.2):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=color, edgecolor='white', linewidth=1.8, zorder=3
    )
    ax.add_patch(rect)
    if subtitle:
        ax.text(cx, cy + 0.17, title,  ha='center', va='center',
                color=C_TEXT, fontsize=fs,  fontweight='bold', zorder=4)
        ax.text(cx, cy - 0.20, subtitle, ha='center', va='center',
                color=C_TEXT, fontsize=fs2, style='italic', zorder=4,
                alpha=0.90)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                color=C_TEXT, fontsize=fs, fontweight='bold', zorder=4)


def arrow_h(ax, x1, x2, y, color=C_ARROW):
    ax.annotate(
        '', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                        mutation_scale=14),
        zorder=2
    )


def arrow_v(ax, x, y1, y2, color=C_ARROW):
    ax.annotate(
        '', xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                        mutation_scale=14),
        zorder=2
    )


def arrow_diag(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                        mutation_scale=14, connectionstyle='arc3,rad=0.0'),
        zorder=2
    )

# ── canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.2))
ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.2)
ax.axis('off')
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── layout constants ──────────────────────────────────────────────────────────
Y_TOP  = 3.80   # main pipeline row
Y_MID  = 2.00   # branch stage row
Y_BOT  = 0.55   # output row

BW = 2.05       # box width  (main)
BH = 0.88       # box height (main)
BW2 = 2.10      # box width  (branch)
BH2 = 0.82      # box height (branch)

# x centres for main row
X = [1.20, 3.65, 6.10, 8.55, 11.00]
# x centres for branches
X_L = 5.50    # polygon branch
X_R = 12.50   # 3D branch

# ── main pipeline boxes ───────────────────────────────────────────────────────
box(ax, X[0], Y_TOP, BW, BH, 'Raw LAS Files',      color=C_INPUT)
box(ax, X[1], Y_TOP, BW, BH, 'Spatial Tiling',     '250 m × 250 m',  color=C_PIPE)
box(ax, X[2], Y_TOP, BW, BH, 'KNN Encoding',       '7-ch · 512×512', color=C_PIPE)
box(ax, X[3], Y_TOP, BW, BH, 'U-Net Segmentation', 'ResNet-34',      color=C_PIPE)
box(ax, X[4], Y_TOP, BW, BH, 'Tile Stitching',                        color=C_FORK)

# ── branch boxes ─────────────────────────────────────────────────────────────
box(ax, X_L, Y_MID, BW2, BH2, 'Polygon Extraction', 'binary mask → contours', color=C_POLY)
box(ax, X_R, Y_MID, BW2, BH2, '3D Back-Projection', 'nearest-neighbour interp.', color=C_CLASS)

# ── output boxes ─────────────────────────────────────────────────────────────
box(ax, X_L, Y_BOT, BW2, 0.74, 'Building Footprints\n(ESRI Shapefile)', color=C_OUT_P, fs=8.0)
box(ax, X_R, Y_BOT, BW2, 0.74, 'Classified LAS\n(per-point labels + RGB)', color=C_OUT_C, fs=8.0)

# ── arrows: main row ─────────────────────────────────────────────────────────
gap = 0.05
for i in range(len(X) - 1):
    arrow_h(ax, X[i] + BW/2 + gap, X[i+1] - BW/2 - gap, Y_TOP)

# ── fork: from Stitching down to a fork point ────────────────────────────────
fork_y  = Y_TOP - BH / 2 - 0.10   # just below last main box
fork_x  = X[4]

ax.plot([fork_x, fork_x], [Y_TOP - BH/2, fork_y - 0.30],
        color=C_ARROW, lw=1.8, zorder=2)

# horizontal line spanning both branches
ax.plot([X_L, X_R], [fork_y - 0.30, fork_y - 0.30],
        color=C_ARROW, lw=1.8, zorder=2)

# vertical drops to branch boxes
arrow_v(ax, X_L, fork_y - 0.30, Y_MID + BH2/2 + 0.04)
arrow_v(ax, X_R, fork_y - 0.30, Y_MID + BH2/2 + 0.04)

# ── arrows: branch → output ──────────────────────────────────────────────────
arrow_v(ax, X_L, Y_MID - BH2/2 - 0.04, Y_BOT + 0.37 + 0.04)
arrow_v(ax, X_R, Y_MID - BH2/2 - 0.04, Y_BOT + 0.37 + 0.04)

# ── mode labels ──────────────────────────────────────────────────────────────
ax.text(X_L, fork_y - 0.12, 'Polygon mode',
        ha='center', va='center', fontsize=7.5,
        color=C_POLY, fontweight='bold', zorder=4)
ax.text(X_R, fork_y - 0.12, '3D classification mode',
        ha='center', va='center', fontsize=7.5,
        color=C_CLASS, fontweight='bold', zorder=4)

# ── step labels (small numbers below boxes) ──────────────────────────────────
labels = ['①', '②', '③', '④', '⑤']
for i, (xi, lbl) in enumerate(zip(X, labels)):
    ax.text(xi, Y_TOP - BH/2 - 0.18, lbl,
            ha='center', va='top', fontsize=8, color='#78909C', zorder=4)

# ── title / caption ───────────────────────────────────────────────────────────
ax.text(8.0, 5.00,
        'Figure 1. Overview of the Proposed Pipeline',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#263238', zorder=4)

# ── save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, 'figure1_pipeline.png')
svg_path = os.path.join(out_dir, 'figure1_pipeline.svg')

plt.savefig(png_path, dpi=300, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')
plt.savefig(svg_path, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')

print(f'Saved:\n  {png_path}\n  {svg_path}')
plt.show()
