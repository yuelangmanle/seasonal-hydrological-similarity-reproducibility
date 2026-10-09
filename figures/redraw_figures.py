#!/usr/bin/env python3
"""Render the three manuscript figures from the supplied source tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',
    'ps.fonttype': 42,
    'savefig.dpi': 1000,
    'savefig.bbox': None,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

C_PROPOSED = '#00A087'
C_FDC      = '#3C5488'
C_TIMING   = '#E64B35'
C_OT       = '#8491B4'
C_ANNUAL   = '#999999'
C_BG_LIGHT = '#F0F7F6'
C_BG_BLUE  = '#EEF1F7'

METHOD_COLORS = {
    'Monthly conditional W2': C_PROPOSED,
    'Conventional FDC': C_FDC,
    'FDC + allocation': C_TIMING,
    'Original OT + FDC': C_OT,
    'Annual log-flow control': C_ANNUAL,
}
METHOD_MARKERS = {
    'Monthly conditional W2': 'D',
    'Conventional FDC': 'o',
    'FDC + allocation': 's',
    'Original OT + FDC': '^',
    'Annual log-flow control': 'p',
}

DATA = Path(__file__).resolve().parent / "figure_data"
OUT = Path(__file__).resolve().parent / "redrawn_figures"
OUT.mkdir(exist_ok=True)

fig1a = pd.read_csv(DATA / "figure2a_overall_errors.csv")
fig1b = pd.read_csv(DATA / "figure2b_attribute_differences.csv")
fig1_pair = pd.read_csv(DATA / "overall_paired_differences.csv")
fig2a_rel = pd.read_csv(DATA / "figure3a_regional_improvement.csv")
fig2b = pd.read_csv(DATA / "figure3b_donor_counts.csv")
fig2c = pd.read_csv(DATA / "figure3c_quantile_resolution.csv")
fig3_diag = pd.read_csv(DATA / "figure1_seasonal_diagnostics.csv")

def finish_figure(fig):
    """Use a common print-safe text size without changing plotted values."""
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if text.get_text().strip():
            text.set_fontsize(max(text.get_fontsize(), 9.5))
    # Native figure width is exactly the journal's two-column width.
    width, height = fig.get_size_inches()
    fig.set_size_inches(178 / 25.4, height * (178 / 25.4) / width)

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(length=0, colors='#444444')

# ----------------------------------------------------------

# ----------------------------------------------------------
fig = plt.figure(figsize=(7.0, 5.2))
fig.subplots_adjust(left=0.23, right=0.985, top=0.84, bottom=0.14)
gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15], wspace=0.32)


gs_left = gs[0].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.5)
ax1a = fig.add_subplot(gs_left[0])
ax1c = fig.add_subplot(gs_left[1])

df = fig1a.sort_values('mean_normalised_MAE', ascending=True).reset_index(drop=True)
y_pos = np.arange(len(df))[::-1]
for i, row in df.iterrows():
    y = y_pos[i]
    c = METHOD_COLORS[row['display_label']]
    m = METHOD_MARKERS[row['display_label']]
    lw = 2.4 if row['display_label'] == 'Monthly conditional W2' else 1.6
    ms = 9 if row['display_label'] == 'Monthly conditional W2' else 7
    ax1a.plot([row['ci95_low'], row['ci95_high']], [y, y], color=c, linewidth=lw, solid_capstyle='butt', alpha=0.9, zorder=3)
    ax1a.plot(row['mean_normalised_MAE'], y, marker=m, color=c, markersize=ms, zorder=5,
              markeredgecolor='white', markeredgewidth=1.2)
    ax1a.text(row['ci95_high'] + 0.008, y, f"{row['mean_normalised_MAE']:.3f}",
              va='center', ha='left', fontsize=7.5, color=c, fontweight='bold')

ax1a.axhspan(y_pos[0]-0.35, y_pos[0]+0.35, facecolor=C_BG_LIGHT, alpha=0.5, zorder=0)
ax1a.set_yticks(y_pos)
ax1a.set_yticklabels(df['display_label'].replace({'Monthly conditional W2':'Conditional W2','Conventional FDC':'FDC','Original OT + FDC':'OT + FDC','Annual log-flow control':'Annual control'}), fontsize=8)
ax1a.set_xlim(0.24, 0.62)
ax1a.set_xlabel('Mean normalised MAE\n(lower is better)', fontsize=7.5, color='#333333', labelpad=2)
ax1a.text(0, 1.02, '485 records · 10 donors', transform=ax1a.transAxes, fontsize=6.8, color='#999999')
style_ax(ax1a)

pair_labels = ['vs FDC', 'vs FDC+alloc.', 'vs OT+FDC']
pair_colors = [C_FDC, C_TIMING, C_OT]
pair_markers = ['o', 's', '^']
y_pair = np.arange(3)[::-1]
for i, (_, row) in enumerate(fig1_pair.iterrows()):
    y = y_pair[i]
    c = pair_colors[i]
    m = pair_markers[i]
    ax1c.plot([row['ci95_low'], row['ci95_high']], [y, y], color=c, linewidth=2.0, solid_capstyle='butt', zorder=3)
    ax1c.plot(row['difference_conditional_minus_comparator'], y, marker=m, color=c,
              markersize=7, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
ax1c.axvline(0, color='#333333', linewidth=0.9)
ax1c.set_yticks(y_pair)
ax1c.set_yticklabels(pair_labels, fontsize=7.5)
ax1c.set_xlim(-0.22, 0.01)
ax1c.set_xlabel('ΔMAE / donor IQR\n(negative favours W2)', fontsize=7.5, color='#333333', labelpad=2)
ax1c.set_title('Paired difference (overall)', loc='left', fontweight='bold', pad=8, color='#222222', fontsize=8.5)
style_ax(ax1c)


gs_right = gs[1].subgridspec(2, 2, hspace=0.65, wspace=0.36)
attrs = ['Baseflow index', 'Flashiness', 'Runoff ratio', 'Snow fraction']
comparators = [
    ('Conventional FDC', C_FDC, 'o'),
    ('FDC + allocation', C_TIMING, 's'),
    ('Original OT + FDC', C_OT, '^'),
]
x_comp = np.arange(3)

for idx, attr in enumerate(attrs):
    ax = fig.add_subplot(gs_right[idx // 2, idx % 2])
    for j, (comp, c, m) in enumerate(comparators):
        sub = fig1b[(fig1b['attribute'] == attr) & (fig1b['display_label'] == comp)]
        if len(sub):
            r = sub.iloc[0]
            ax.bar(j, r['difference_conditional_minus_comparator'], width=0.6,
                   color=c, alpha=0.85, zorder=3, edgecolor='white', linewidth=0.8)
            err_low = r['difference_conditional_minus_comparator'] - r['ci95_low']
            err_high = r['ci95_high'] - r['difference_conditional_minus_comparator']
            ax.errorbar(j, r['difference_conditional_minus_comparator'],
                        yerr=[[err_low], [err_high]], fmt='none', ecolor='#555555',
                        elinewidth=1.0, capsize=2.5, zorder=4)
    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xticks(x_comp)
    ax.set_xticklabels(['FDC', '+alloc.', 'OT+FDC'], fontsize=6.8, color='#555555', rotation=35, ha='right')
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.42, 0.06)
    ax.set_title(attr, loc='center', fontsize=8, fontweight='bold', color='#333333', pad=3)
    style_ax(ax)
    if idx % 2 == 0:
        ax.set_ylabel('ΔMAE / donor IQR', fontsize=7, color='#555555', labelpad=2)
    else:
        ax.set_yticklabels([])
    if idx == 0:
        ax.text(0.98, 0.12, '↓ favours W2', transform=ax.transAxes, fontsize=7.0,
                color=C_PROPOSED, va='bottom', ha='right', style='italic')

fig.text(0.23, 0.975, 'a   Overall error and\n     paired differences', fontweight='bold', fontsize=9.5, color='#222222', va='top')
fig.text(0.61, 0.975, 'b   Attribute-level\n     paired improvements', fontweight='bold', fontsize=9.5, color='#222222', va='top')

finish_figure(fig)
fig.savefig(OUT / "Figure2_internal_comparison.png", dpi=1000)
fig.savefig(OUT / "Figure2_internal_comparison.pdf")
fig.savefig(OUT / "Figure2_internal_comparison.svg")
fig.savefig(OUT / "Figure2_internal_comparison.tiff", dpi=1000, pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print("Fig1 done")

# ----------------------------------------------------------

# ----------------------------------------------------------
fig = plt.figure(figsize=(7.0, 7.0))
fig.subplots_adjust(left=0.14, right=0.98, top=0.91, bottom=0.13)
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.42, wspace=0.30)


ax_a = fig.add_subplot(gs[0, :])
regions = ['CAMELS-US', 'CAMELS-BR', 'CAMELS-CL', 'HYSETS', 'LamaH-CE']
comps = ['Conventional FDC', 'FDC + allocation', 'Original OT + FDC']
comp_colors = [C_FDC, C_TIMING, C_OT]
comp_labels = ['FDC', 'FDC+alloc.', 'OT+FDC']

n_reg = len(regions)
x_reg = np.arange(n_reg)
bar_w = 0.25

for j, (comp, c, clab) in enumerate(zip(comps, comp_colors, comp_labels)):
    vals = []
    for reg in regions:
        row = fig2a_rel[(fig2a_rel['region_label'] == reg) & (fig2a_rel['comparator_label'] == comp)]
        vals.append(row.iloc[0]['relative_improvement_percent'])
    bars = ax_a.bar(x_reg + (j - 1) * bar_w, vals, width=bar_w, color=c, alpha=0.85,
                    label=clab, edgecolor='white', linewidth=0.8, zorder=3)
    
    for i, v in enumerate(vals):
        offset = 0.8 if v >= 0 else -0.8
        va = 'bottom' if v >= 0 else 'top'
        ax_a.text(x_reg[i] + (j - 1) * bar_w, v + offset, f"{v:+.1f}",
                  ha='center', va=va, fontsize=7, color=c, fontweight='bold')

ax_a.axhline(0, color='#333333', linewidth=1.0, zorder=2)
ax_a.set_xticks(x_reg)
ax_a.set_xticklabels([f"{r}\n(n={fig2a_rel[fig2a_rel['region_label']==r]['n_test_records'].iloc[0]})" for r in regions],
                     fontsize=8, color='#333333')
ax_a.set_ylim(-22, 32)
ax_a.set_ylabel('Relative reduction in aggregate error (%)', fontsize=8.5, color='#333333', labelpad=4)
ax_a.set_title('a   Regional transfer boundary  (positive favours conditional W2)',
               loc='left', fontweight='bold', pad=10, color='#222222', fontsize=10)
ax_a.legend(fontsize=8, frameon=False, loc='upper left', ncol=3,
            handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.0, 1.02))
style_ax(ax_a)
ax_a.grid(axis='y', alpha=0.12, linewidth=0.4)

ax_a.axhspan(0, 32, facecolor=C_BG_LIGHT, alpha=0.3, zorder=0)
ax_a.axhspan(-22, 0, facecolor='#FDF0EF', alpha=0.3, zorder=0)


ax_b = fig.add_subplot(gs[1, 0])
methods_b = [
    ('Monthly conditional W2', C_PROPOSED, 'D', '-'),
    ('FDC + allocation', C_TIMING, 's', '-'),
    ('Original OT + FDC', C_OT, '^', '-'),
    ('Conventional FDC', C_FDC, 'o', '--'),
]
for lab, c, m, ls in methods_b:
    sub = fig2b[fig2b['display_label'] == lab].sort_values('number_of_donors')
    lw = 2.2 if lab == 'Monthly conditional W2' else 1.4
    ax_b.plot(sub['number_of_donors'], sub['mean_normalised_MAE'],
              marker=m, color=c, linestyle=ls, linewidth=lw, markersize=6,
              markeredgecolor='white', markeredgewidth=0.8, label={'Monthly conditional W2':'Conditional W2','Original OT + FDC':'OT + FDC','Conventional FDC':'FDC'}.get(lab,lab), alpha=0.9)
ax_b.set_xlabel('Number of donors', labelpad=4, color='#333333')
ax_b.set_ylabel('Mean normalised error', labelpad=4, color='#333333')
ax_b.set_xticks([5, 10, 20])
ax_b.set_title('b   Donor-count sensitivity', loc='left', fontweight='bold', pad=10, color='#222222', fontsize=9.5)
ax_b.legend(fontsize=6.8, frameon=False, loc='upper left',
            bbox_to_anchor=(0.02, 0.91), ncol=1,
            handlelength=1.2, handletextpad=0.2, labelspacing=0.3, columnspacing=0.6)
style_ax(ax_b)


ax_c = fig.add_subplot(gs[1, 1])
monthly = fig2c[fig2c['representation'] == 'monthly'].sort_values('number_of_quantiles')
annual = fig2c[fig2c['representation'] == 'annual'].sort_values('number_of_quantiles')

ax_c.plot(monthly['number_of_quantiles'], monthly['mean_normalised_MAE'],
          marker='D', color=C_PROPOSED, linewidth=2.2, markersize=6.5,
          markeredgecolor='white', markeredgewidth=0.8, label='Monthly conditional')
ax_c.plot(annual['number_of_quantiles'], annual['mean_normalised_MAE'],
          marker='o', color=C_ANNUAL, linewidth=1.4, markersize=6.5, linestyle='--',
          markeredgecolor='white', markeredgewidth=0.8, label='Annual control', alpha=0.8)

ax_c.set_xlabel('Quantiles per distribution', labelpad=4, color='#333333')
ax_c.set_ylabel('Mean normalised error', labelpad=4, color='#333333')
ax_c.set_xticks([32, 64, 128])
ax_c.set_title('c   Quantile-resolution control', loc='left', fontweight='bold', pad=10, color='#222222', fontsize=9.5)
ax_c.legend(fontsize=7.5, frameon=False, loc='center right')
style_ax(ax_c)

finish_figure(fig)
fig.savefig(OUT / "Figure3_regional_robustness.png", dpi=1000)
fig.savefig(OUT / "Figure3_regional_robustness.pdf")
fig.savefig(OUT / "Figure3_regional_robustness.svg")
fig.savefig(OUT / "Figure3_regional_robustness.tiff", dpi=1000, pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print("Fig2 done")

# ----------------------------------------------------------

# ----------------------------------------------------------
fig = plt.figure(figsize=(7.0, 6.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.42)

ax_s = fig.add_subplot(gs[0])
ax_s.set_xlim(0, 12)
ax_s.set_ylim(0, 5.5)
ax_s.axis('off')
ax_s.set_title('a   Method construction (schematic)' , loc='left', fontweight='bold', pad=8, color='#222222', fontsize=10)

ax_s.add_patch(Rectangle((0.1, 0.3), 3.6, 4.6, facecolor=C_BG_BLUE, alpha=0.5, zorder=0))
ax_s.add_patch(Rectangle((4.1, 0.3), 3.8, 4.6, facecolor=C_BG_LIGHT, alpha=0.6, zorder=0))
ax_s.add_patch(Rectangle((8.3, 0.3), 3.6, 4.6, facecolor='#FAFAFA', alpha=0.8, zorder=0))

ax_s.text(1.9, 4.65, 'Annual FDC', ha='center', fontsize=10, fontweight='bold', color=C_FDC)
ax_s.text(6.0, 4.65, 'Monthly conditional W2', ha='center', fontsize=10, fontweight='bold', color=C_PROPOSED)
ax_s.text(10.1, 4.65, 'Donor selection', ha='center', fontsize=10, fontweight='bold', color='#555555')

x_fdc = np.linspace(0, 1, 100)
y_fdc = 0.15 + 0.75 * np.exp(-x_fdc * 2.5)
ax_s.plot(0.5 + x_fdc * 2.8, 1.5 + y_fdc * 2.0, color=C_FDC, linewidth=2.8, zorder=3)
ax_s.fill_between(0.5 + x_fdc * 2.8, 1.5, 1.5 + y_fdc * 2.0, color=C_FDC, alpha=0.08, zorder=2)
ax_s.plot([1.2, 2.6], [3.0, 3.0], color=C_TIMING, linewidth=1.5, linestyle='--', alpha=0.6)
ax_s.text(1.9, 3.2, 'month erased', ha='center', fontsize=7.5, color=C_TIMING, style='italic')
ax_s.text(1.9, 0.7, 'Daily records →\nannual distribution', ha='center', fontsize=7, color='#888888')
circ = Circle((1.9, 2.5), 0.22, fill=False, color='#CCCCCC', linewidth=1.8, zorder=4)
ax_s.add_patch(circ)
ax_s.plot([1.74, 2.06], [2.34, 2.66], color='#CCCCCC', linewidth=1.8, zorder=4)

ax_s.annotate('', xy=(4.0, 2.8), xytext=(3.75, 2.8),
              arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=2.0))

months = ['J','F','M','A','M','J','J','A','S','O','N','D']
for mi in range(12):
    col = mi % 4
    row = mi // 4
    mx = 4.4 + col * 0.85
    my = 3.5 - row * 0.85
    xm = np.linspace(0, 1, 30)
    season_factor = 0.6 + 0.4 * np.sin(mi * np.pi / 6 + 0.5)
    ym = 0.08 + 0.25 * season_factor * np.exp(-xm * 2.2)
    ax_s.plot(mx + xm * 0.7, my + ym * 1.2, color=C_PROPOSED, linewidth=1.3, alpha=0.8, zorder=3)
    ax_s.text(mx + 0.35, my - 0.12, months[mi], ha='center', fontsize=6, color='#999999')
ax_s.text(6.0, 0.7, '12 months × 32 quantiles\n(log1p(Q / mean Q))', ha='center', fontsize=7, color='#888888')

ax_s.annotate('', xy=(8.2, 2.8), xytext=(7.95, 2.8),
              arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=2.0))

x_w = np.linspace(0, 1, 50)
y_target = 0.2 + 0.6 * np.exp(-((x_w - 0.35)**2) / 0.04)
y_donor = 0.15 + 0.55 * np.exp(-((x_w - 0.55)**2) / 0.05)
ax_s.plot(8.7 + x_w * 1.2, 3.2 + y_target * 1.0, color=C_PROPOSED, linewidth=2.0, zorder=3)
ax_s.plot(8.7 + x_w * 1.2, 3.2 + y_donor * 1.0, color=C_ANNUAL, linewidth=1.5, linestyle='--', zorder=3)
ax_s.text(10.1, 3.95, 'W2 on monthly\nquantiles', ha='center', fontsize=7.2, color=C_PROPOSED, fontweight='bold')

donor_pts = [(8.8, 1.8), (9.3, 2.2), (9.8, 1.6), (10.5, 2.0), (11.0, 1.7)]
for dx, dy in donor_pts:
    ax_s.plot(dx, dy, 'o', color='#E0E0E0', markersize=8, markeredgecolor='#BBBBBB', markeredgewidth=0.8, zorder=3)
target_x, target_y = 9.6, 1.9
ax_s.plot(target_x, target_y, '*', color=C_PROPOSED, markersize=18, zorder=5,
          markeredgecolor='white', markeredgewidth=0.5)
ax_s.plot([target_x, 9.3], [target_y, 2.2], color=C_PROPOSED, linewidth=1.2, linestyle='--', alpha=0.5, zorder=2)
ax_s.text(10.1, 0.7, 'nearest by W2', ha='center', fontsize=7, color='#888888')


ax_d = fig.add_subplot(gs[1])
predictors = ['FDC', 'composite', 'individual_signatures']
pred_labels = ['FDC', 'Composite', 'Individual\nsignatures']
x = np.arange(len(predictors))

ax_d.axvspan(-0.4, 0.4, facecolor=C_BG_BLUE, alpha=0.4, zorder=0)
ax_d.axvspan(0.6, 1.4, facecolor='#F5F5F5', alpha=0.5, zorder=0)
ax_d.axvspan(1.6, 2.4, facecolor=C_BG_LIGHT, alpha=0.5, zorder=0)

for i, p in enumerate(predictors):
    row_p = fig3_diag[(fig3_diag['scope'] == 'pooled') & (fig3_diag['predictor'] == p)].iloc[0]
    row_w = fig3_diag[(fig3_diag['scope'] == 'within') & (fig3_diag['predictor'] == p)].iloc[0]
    v_p = row_p['knn_test_R2']
    v_w = row_w['knn_test_R2']
    ax_d.plot([i, i], [v_p, v_w], color='#CCCCCC', linewidth=2.0, zorder=1, solid_capstyle='round', linestyle=':')
    ax_d.plot(i, v_p, 'o', color=C_FDC, markersize=11, zorder=5,
              markerfacecolor='white', markeredgewidth=2.0, markeredgecolor=C_FDC)
    ax_d.plot(i, v_w, 'o', color=C_PROPOSED, markersize=11, zorder=5,
              markeredgecolor='white', markeredgewidth=1.5)
    if v_p < 0.03:
        ax_d.text(i, v_p + 0.025, f'{v_p:.2f}', ha='center', va='bottom', fontsize=7.5, color=C_FDC, fontweight='bold')
    else:
        ax_d.text(i, v_p - 0.028, f'{v_p:.2f}', ha='center', va='top', fontsize=7.5, color=C_FDC)
    ax_d.text(i, v_w + 0.018, f'{v_w:.2f}', ha='center', va='bottom', fontsize=7.5, color=C_PROPOSED, fontweight='bold')
    delta = v_w - v_p
    if delta > 0.02:
        ax_d.annotate(f'+{delta:.2f}', xy=(i+0.18, (v_p+v_w)/2), fontsize=7.0,
                      color=C_PROPOSED, style='italic', va='center')

ax_d.set_xticks(x)
ax_d.set_xticklabels(pred_labels, fontsize=9, color='#333333', fontweight='bold')
ax_d.set_ylabel('Test R² of timing-distance\nprediction (25-NN)', fontsize=8, color='#333333', labelpad=4)
ax_d.set_ylim(-0.02, 0.52)
ax_d.set_title('b   Scalar FDC distance weakly predicts\n     seasonal separation', loc='left', fontweight='bold', pad=10, color='#222222', fontsize=10)
# Station separation and open/filled encoding are documented in the caption and legend.
style_ax(ax_d)

legend_d = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=C_FDC,
           markeredgewidth=2, markersize=9, label='pooled (all regions)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_PROPOSED, markeredgecolor='white',
           markersize=9, label='within (same region)'),
]
ax_d.legend(handles=legend_d, loc='upper left', frameon=False, fontsize=7.5,
            handletextpad=0.3, bbox_to_anchor=(0.0, 1.02))

finish_figure(fig)
fig.savefig(OUT / "Figure1_diagnosis_construction.png", dpi=1000)
fig.savefig(OUT / "Figure1_diagnosis_construction.pdf")
fig.savefig(OUT / "Figure1_diagnosis_construction.svg")
fig.savefig(OUT / "Figure1_diagnosis_construction.tiff", dpi=1000, pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print("Fig3 done")
