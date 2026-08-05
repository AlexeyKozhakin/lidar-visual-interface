"""
Figure 3 — Dataset samples panel (3 splits × 4 columns).
Columns: CloudCompare placeholder | img_rgb | img_features | img_class
Rows:    Train | Val | Test
Run: python paper_arxiv/figures/generate_dataset_samples_figure.py
Output: paper_arxiv/figures/figure3_dataset_samples.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import os

# ── samples to show ───────────────────────────────────────────────────────────
ROWS = [
    dict(split='train', sample='Austin_250_250',       label='Train', cc_path=None),
    dict(split='val',   sample='FortWorth_250_250',    label='Val',   cc_path=None),
    dict(split='test',  sample='SanFrancisco_500_500', label='Test',  cc_path='paper_arxiv/img/SanFrancisco_500_500.png'),
]

BASE = 'data/stpls3d_ready'

# ── colours ───────────────────────────────────────────────────────────────────
C_BG        = '#F8F9FA'
C_HEADER    = '#1565C0'
C_ROW_TRAIN = '#2E7D32'
C_ROW_VAL   = '#6A1B9A'
C_ROW_TEST  = '#BF360C'
ROW_COLORS  = [C_ROW_TRAIN, C_ROW_VAL, C_ROW_TEST]

COL_HEADERS = ['CloudCompare\n(3D view)', 'img_rgb\n(RGB projection)',
               'img_features\n(KNN channels)', 'img_class\n(ground truth)']

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor(C_BG)

gs = GridSpec(3, 4, figure=fig,
              left=0.10, right=0.98, top=0.91, bottom=0.04,
              wspace=0.05, hspace=0.12)

for r, row in enumerate(ROWS):
    split  = row['split']
    sample = row['sample']
    color  = ROW_COLORS[r]

    # load real images
    rgb  = np.array(Image.open(f'{BASE}/{split}/img_rgb/{sample}.png'))
    feat = np.array(Image.open(f'{BASE}/{split}/img_features/{sample}.png'))
    cls  = np.array(Image.open(f'{BASE}/{split}/img_class/{sample}.png'))

    cc = np.array(Image.open(row['cc_path'])) if row['cc_path'] else None
    images = [cc, rgb, feat, cls]   # cc=None → placeholder

    for c, img in enumerate(images):
        ax = fig.add_subplot(gs[r, c])
        ax.set_aspect('equal')

        if img is None:
            # black placeholder with tile name
            ax.set_facecolor('black')
            ax.text(0.5, 0.58, sample.replace('_', '\n'),
                    ha='center', va='center', transform=ax.transAxes,
                    color='white', fontsize=7.5, fontweight='bold',
                    linespacing=1.6)
            ax.text(0.5, 0.28, '[ CloudCompare\n  screenshot ]',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#888888', fontsize=6.5, style='italic',
                    linespacing=1.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(1.5)
        else:
            ax.imshow(img, interpolation='nearest')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    # row label on the left
    fig.text(0.01, gs[r, 0].get_position(fig).y0 +
             gs[r, 0].get_position(fig).height / 2,
             row['label'],
             ha='left', va='center', fontsize=10,
             fontweight='bold', color=color,
             rotation=90)

# column headers
for c, title in enumerate(COL_HEADERS):
    pos = gs[0, c].get_position(fig)
    fig.text(pos.x0 + pos.width / 2, 0.935,
             title, ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color=C_HEADER,
             linespacing=1.5)

# caption
fig.text(0.5, 0.005,
         'Figure 3.  Representative samples from the train, validation, and test splits of the prepared STPLS3D dataset. '
         'Each row shows one 250 m tile from a different city: Austin (train), Fort Worth (val), San Francisco (test). '
         'Columns: 3D point-cloud visualization in CloudCompare (placeholder), RGB projection (img_rgb), '
         'KNN-encoded feature image (img_features), and semantic ground-truth mask (img_class).',
         ha='center', va='bottom', fontsize=7, color='#37474F', style='italic')

# ── save ─────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'figure3_dataset_samples.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=C_BG)
print(f'Saved: {out}')
plt.show()
