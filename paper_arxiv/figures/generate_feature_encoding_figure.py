"""
Figure for Section 3.3 — 3D-to-2D feature encoding visualization.
Shows: img_rgb → z_mean / n_z / n_r (individual channels) → img_features → img_class
Run: python paper_arxiv/figures/generate_feature_encoding_figure.py
Output: paper_arxiv/figures/figure_feature_encoding.png  (300 dpi)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import os

BASE = 'data/stpls3d_ready/test'
SAMPLE = 'Memphis_500_250'

# ── load data ─────────────────────────────────────────────────────────────────
tensor   = np.load(f'{BASE}/tensors/{SAMPLE}.npy')          # (512,512,7)
img_rgb  = np.array(Image.open(f'{BASE}/img_rgb/{SAMPLE}.png'))
img_feat = np.array(Image.open(f'{BASE}/img_features/{SAMPLE}.png'))
img_cls  = np.array(Image.open(f'{BASE}/img_class/{SAMPLE}.png'))

# extract individual channels and normalise to [0,1] for display
def norm(ch):
    mn, mx = ch.min(), ch.max()
    return (ch - mn) / (mx - mn + 1e-8)

z_mean = norm(tensor[..., 0])   # terrain-corrected elevation
n_z    = norm(tensor[..., 1])   # vertical normal proxy
n_r    = norm(tensor[..., 2])   # radial normal proxy

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 3.8))
fig.patch.set_facecolor('#F8F9FA')

# 6 image slots + narrow spacer columns for arrows
# ratios: img img gap img img img gap img img
gs = GridSpec(1, 6, figure=fig,
              left=0.02, right=0.98, top=0.80, bottom=0.08,
              wspace=0.06)

axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

panels = [
    (img_rgb,  'img_rgb',       '(a) RGB projection\n(point cloud input)',      None),
    (z_mean,   'z_mean',        '(b) $z_{\\mathrm{mean}}$\nterrain-corrected elev.', 'gray'),
    (n_z,      'n_z',           '(c) $n_z$\nvertical normal proxy',             'gray'),
    (n_r,      'n_r',           '(d) $n_r$\nradial normal proxy',               'gray'),
    (img_feat, 'img_features',  '(e) Feature image\n($z$, $n_z$, $n_r$ → RGB)', None),
    (img_cls,  'img_class',     '(f) Ground truth\nsemantic mask',              None),
]

C_TITLE = '#1A237E'   # deep blue for titles

for ax, (img, key, title, cmap) in zip(axes, panels):
    if cmap is not None:
        ax.imshow(img, cmap=cmap, interpolation='nearest')
    else:
        ax.imshow(img, interpolation='nearest')
    ax.axis('off')

    # title above each panel
    ax.set_title(title, fontsize=7.5, color=C_TITLE, pad=4,
                 fontweight='bold' if key in ('img_rgb', 'img_features', 'img_class') else 'normal',
                 linespacing=1.5)

    # thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#90A4AE')
        spine.set_linewidth(0.8)

# ── group labels ─────────────────────────────────────────────────────────────
# "Input", "KNN Feature Channels", "Output"
def bracket(fig, ax_left, ax_right, label, color, y=0.85):
    x0 = ax_left.get_position().x0
    x1 = ax_right.get_position().x1
    xm = (x0 + x1) / 2
    fig.text(xm, y, label, ha='center', va='bottom',
             fontsize=8.5, color=color, fontweight='bold',
             transform=fig.transFigure)
    # underline
    fig.add_artist(plt.Line2D([x0 + 0.005, x1 - 0.005], [y - 0.015, y - 0.015],
                              transform=fig.transFigure,
                              color=color, linewidth=1.5, linestyle='-'))

bracket(fig, axes[0], axes[0], 'Input',                '#37474F', y=0.87)
bracket(fig, axes[1], axes[3], 'KNN Feature Channels', '#1565C0', y=0.87)
bracket(fig, axes[4], axes[5], 'Model I/O',            '#2E7D32', y=0.87)

# ── arrows between groups ────────────────────────────────────────────────────
def inter_arrow(fig, ax_left, ax_right, color='#546E7A'):
    x0 = ax_left.get_position().x1
    x1 = ax_right.get_position().x0
    y  = (ax_left.get_position().y0 + ax_left.get_position().y1) / 2
    ax_left.annotate('', xy=(x1 + 0.003, y), xytext=(x0 - 0.003, y),
                     xycoords='figure fraction', textcoords='figure fraction',
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                     mutation_scale=10))

inter_arrow(fig, axes[0], axes[1])   # input → channels
inter_arrow(fig, axes[3], axes[4])   # channels → feature image

# ── main caption ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.02,
         'Figure X.  3D-to-2D feature encoding for a single 250 m tile (SanFrancisco_500_500). '
         'The terrain-corrected elevation ($z_{\\mathrm{mean}}$), vertical normal proxy ($n_z$), '
         'and radial normal proxy ($n_r$) are each extracted by KNN aggregation and\n'
         'combined into a three-channel feature image used as input to the U-Net model.',
         ha='center', va='bottom', fontsize=7.2, color='#37474F',
         style='italic', wrap=True)

# ── save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
out_png = os.path.join(out_dir, 'figure_feature_encoding.png')
out_svg = os.path.join(out_dir, 'figure_feature_encoding.svg')

plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
plt.savefig(out_svg, bbox_inches='tight', facecolor='#F8F9FA')
print(f'Saved:\n  {out_png}\n  {out_svg}')
plt.show()
