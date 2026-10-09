"""Figure 4: climate support, component ablation and temperate donor support.

All values are read from the released climate_process CSVs. Individual paired
95% station-group bootstrap intervals are shown, without multiplicity adjustment.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'reproduction/climate_process'
OUT = ROOT / 'climate_process'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.3,
})
scores = pd.read_csv(DATA / 'scores.csv')
paired = pd.read_csv(DATA / 'paired_subgroups.csv')
attrs = pd.read_csv(DATA / 'attribute_contrasts.csv')
members = pd.read_csv(DATA / 'climate_membership.csv')
counts = pd.read_csv(DATA / 'sample_counts.csv')
TEAL, CORAL, BLUE, GREY = '#007F82', '#C75A48', '#4678AC', '#72777E'
fig = plt.figure(figsize=(178 / 25.4, 185 / 25.4), facecolor='white')
gs = fig.add_gridspec(2, 2, left=.14, right=.98, top=.90, bottom=.18,
                     hspace=.65, wspace=.68)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
for ax, label, title in zip(axes, 'abcd', [
        'Temperate sample', 'Location-shape ablation',
        'Donor-pool dependence', 'Snow-fraction matching']):
    ax.set_title(title, loc='left', pad=24, weight='bold')
    ax.text(-.18, 1.20, label, transform=ax.transAxes, weight='bold', size=11)
    ax.tick_params(length=3, color='#777777')
    ax.set_axisbelow(True)

# a: population counts have no sampling uncertainty; every retained record is used.
ax = axes[0]
datasets = ['camels', 'camelsbr', 'camelscl', 'hysets', 'lamah']
labels = ['CAMELS-US', 'CAMELS-BR', 'CAMELS-CL', 'HYSETS', 'LamaH-CE']
table = counts.groupby(['dataset', 'band']).records.sum().unstack(fill_value=0)
cool = table.loc[datasets, 'cool'].to_numpy()
warm = table.loc[datasets, 'warm'].to_numpy()
ypos = np.arange(5)
ax.barh(ypos, cool, color=BLUE, height=.57, label='Cool temperate')
ax.barh(ypos, warm, left=cool, color=CORAL, height=.57, label='Warm temperate')
for y, total in zip(ypos, cool + warm):
    ax.text(total + 6, y, str(total), va='center', size=8)
ax.set_yticks(ypos, labels)
ax.invert_yaxis()
ax.set_xlim(0, max(cool + warm) * 1.22)
ax.set_xlabel('Retained catchment records')
ax.grid(axis='x', color='#E7EAED', lw=.6)
ax.legend(loc='upper left', bbox_to_anchor=(-.01, -.25), frameon=False)
ax.text(0, 1.03, '787 records; 145 in the test partition', transform=ax.transAxes,
        size=7.5, color=GREY)

# b: unmixed ablations isolate the components without changing FDC weights.
ax = axes[1]
rows = [('pooled_unmixed', 'quantile_location', 'Pooled: location', CORAL),
        ('pooled_unmixed', 'quantile_shape', 'Pooled: shape', BLUE),
        ('temperate_only_unmixed', 'quantile_location', 'Temperate: location', CORAL),
        ('temperate_only_unmixed', 'quantile_shape', 'Temperate: shape', BLUE)]
for i, (design, method, label, color) in enumerate(rows):
    row = paired[(paired.design == design) & (paired.subgroup == 'all') &
                 (paired.method == method)].iloc[0]
    ax.errorbar(row.monthly_minus_method, i,
                xerr=[[row.monthly_minus_method - row.ci_low],
                      [row.ci_high - row.monthly_minus_method]],
                fmt='o', color=color, capsize=3, markersize=5)
ax.axvline(0, color=GREY, lw=.8, ls='--')
ax.set_yticks(range(4), [r[2] for r in rows])
ax.invert_yaxis()
ax.set_xlim(-.065, .025)
ax.set_xticks([-.06, -.03, 0])
ax.set_xlabel('Full monthly minus component error')
ax.text(0, 1.03, 'Unmixed methods; individual 95% intervals',
        transform=ax.transAxes, size=7.5, color=GREY)
ax.text(.03, -.28, 'Negative: full monthly is better', transform=ax.transAxes,
        size=7.5, color=TEAL)
ax.grid(axis='x', color='#E7EAED', lw=.6)

# c: fixed original-donor IQRs keep the scale identical across donor designs.
# These are point scores on one shared test set, not replicate means.
ax = axes[2]
designs = ['temperate_global_donors', 'temperate_only', 'temperate_200km']
for method, label, color, marker in [
        ('FDC', 'FDC', GREY, 's'), ('monthly', 'Monthly, calibrated', TEAL, 'o'),
        ('monthly_regime', 'Regime + FDC', CORAL, '^')]:
    vals = [scores[(scores.design == d) & (scores.method == method)].iloc[0].score_original_scale
            for d in designs]
    ax.plot(range(3), vals, marker=marker, color=color, label=label, markersize=4.5)
# Convert raw unmixed errors to the original donor scale from the attribute table.
pooled_scale = np.array(json.loads(scores[(scores.design == 'pooled') &
                                        (scores.method == 'monthly')].iloc[0].attribute_iqr))
pure = []
for design in designs:
    block = attrs[(attrs.design == design + '_unmixed') & (attrs.subgroup == 'all') &
                  (attrs.method == 'monthly')].set_index('attribute')
    vals = block.loc[['BFI', 'flashiness', 'runoff_ratio', 'snow_fraction'], 'mae'].to_numpy()
    pure.append(float(np.mean(vals / pooled_scale)))
ax.plot(range(3), pure, marker='D', color=TEAL, ls=':', mfc='white',
        label='Monthly, unmixed', markersize=5)
ax.set_xticks(range(3), ['Global\ndonors', 'Temperate\ndonors', 'Temperate\n>200 km'])
ax.set_ylabel('Error on original donor-IQR scale')
ax.set_ylim(.22, .48)
ax.grid(axis='y', color='#E7EAED', lw=.6)
ax.text(0, 1.03, 'Same 145 temperate test records', transform=ax.transAxes,
        size=7.5, color=GREY)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.045, .005),
           frameon=False, ncol=2, columnspacing=1.2, handlelength=1.7)

# d: the attribute-level evidence for keeping monthly location in temperate donors.
ax = axes[3]
for i, (design, label) in enumerate([
        ('pooled_unmixed', 'Pooled'),
        ('temperate_global_donors_unmixed', 'Temperate / global'),
        ('temperate_only_unmixed', 'Temperate / local')]):
    row = attrs[(attrs.design == design) & (attrs.subgroup == 'all') &
                (attrs.method == 'quantile_shape') &
                (attrs.attribute == 'snow_fraction')].iloc[0]
    ax.errorbar(row.monthly_minus_method, i,
                xerr=[[row.monthly_minus_method - row.ci_low],
                      [row.ci_high - row.monthly_minus_method]],
                fmt='o', color=TEAL, capsize=3, markersize=5)
ax.axvline(0, color=GREY, lw=.8, ls='--')
ax.set_yticks(range(3), ['Pooled', 'Temperate / global', 'Temperate / restricted'])
ax.invert_yaxis()
ax.set_xlim(-.017, .006)
ax.set_xticks([-.015, -.01, -.005, 0])
ax.tick_params(axis='x', labelsize=7)
ax.set_xlabel('Full monthly minus shape-only MAE')
ax.text(0, 1.03, 'Unmixed methods; individual 95% intervals',
        transform=ax.transAxes, size=7.5, color=GREY)
ax.text(.03, -.28, 'Negative: keeping location helps', transform=ax.transAxes,
        size=7.5, color=TEAL)
ax.grid(axis='x', color='#E7EAED', lw=.6)

fig.canvas.draw()
for ext in ['pdf', 'svg']:
    fig.savefig(OUT / ('Figure4_climate_process.' + ext))
fig.savefig(OUT / 'Figure4_climate_process.png', dpi=300)
fig.savefig(OUT / 'Figure4_climate_process_178mm.tiff', dpi=1000,
            pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)
print('Figure 4 written at 178 x 185 mm')
