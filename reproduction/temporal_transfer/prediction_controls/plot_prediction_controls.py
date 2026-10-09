"""Render temporal prediction controls and held-out donor-gain validation.

Run beside the released JSON result files, or pass --data-dir and --out-dir.
All plotted summaries are read from the supplied analysis outputs. The
supplement shows paired physical-site bootstrap intervals without refitting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WINDOWS = ['3yr', '5yr', '10yr', '5yr_shifted']
DISPLAY = ['3 yr', '5 yr', '10 yr', '5 yr shifted']
COLORS = {'Persistence': '#626873', 'Monthly donor': '#357C79',
          'Monthly blend': '#AF6145', 'Annual-FDC blend': '#626873',
          'Allocation blend': '#96875C'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
})


def label(ax, letter, title):
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                xytext=(-18, 12), textcoords='offset points',
                fontsize=11, weight='bold', ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=12, weight='normal')


def save(fig, out_dir, name):
    """Measure rendered axes and preserve physical alignment evidence."""
    fig.canvas.draw()
    rectangles = []
    for ax in fig.axes:
        p = ax.get_position()
        rectangles.append([p.x0*fig.get_figwidth()*72,
                           p.y0*fig.get_figheight()*72,
                           p.x1*fig.get_figwidth()*72,
                           p.y1*fig.get_figheight()*72])
    # Both figures deliberately use equal-width, vertically aligned plot areas.
    assert max(r[0] for r in rectangles)-min(r[0] for r in rectangles) <= 1.5
    assert max(r[2] for r in rectangles)-min(r[2] for r in rectangles) <= 1.5
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        (out_dir/f'{name}.alignment.json').write_text(json.dumps({
            'status': 'PASS', 'backend': 'matplotlib', 'tolerance_pt': 1.5,
            'plot_rectangles_pt': rectangles,
            'checks': ['equal left edges', 'equal right edges'],
        }, indent=2))
    else:
        require_matplotlib_panel_alignment(
            fig, json_out=out_dir/f'{name}.alignment.json',
            overlay_svg=out_dir/f'{name}.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # Fixed-size export preserves the stated 178-mm width.
    fig.savefig(out_dir/f'{name}.pdf', facecolor='white')
    fig.savefig(out_dir/f'{name}.svg', facecolor='white')
    fig.savefig(out_dir/f'{name}.png', dpi=600, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    read = lambda fn: json.loads((args.data_dir/fn).read_text())
    blend = read('blend_results.json')
    control = read('blend_controls_results.json')
    prediction = read('results.json')
    rows = []
    x = np.arange(4)
    fig, axes = plt.subplots(2, 1, figsize=(7.007874, 5.118110))
    fig.subplots_adjust(left=.14, right=.98, top=.85, bottom=.10, hspace=.72)
    ax = axes[0]
    label(ax, 'a', 'Target persistence and donor-assisted prediction')
    for method, key, dx, marker in [
            ('Persistence', 'persistence_error', -.19, 'o'),
            ('Monthly donor', 'donor_error', 0, 's'),
            ('Monthly blend', 'blend_error', .19, 'D')]:
        vals = np.array([blend[w]['four_labels'][key] for w in WINDOWS])
        ax.plot(x+dx, vals, linestyle='none', marker=marker, ms=5.2,
                color=COLORS[method], label=method, zorder=3)
        for w, v in zip(WINDOWS, vals):
            rows.append({'figure': '4', 'panel': 'a', 'setting': w,
                         'series': method, 'estimate': v,
                         'low': '', 'high': ''})
    ax.set_xlim(-.58, 3.58)
    ax.set_ylim(0, .36)
    ax.set_yticks([0, .1, .2, .3])
    ax.set_ylabel('Mean normalized absolute error')
    ax.set_xticks(x, [f'{d}\nα = {blend[w]["alpha_selected"]:.2f}'
                         for d,w in zip(DISPLAY, WINDOWS)])
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
              bbox_to_anchor=(.57, .99), ncol=3,
              columnspacing=1.9, handletextpad=.5)
    ax.yaxis.grid(True, color='#E3E5E6', linewidth=.5)
    ax.set_axisbelow(True)
    ax = axes[1]
    label(ax, 'b', 'Observed and predicted benefit of monthly donor retrieval')
    domains = ['primary', 'camels', 'camelsbr', 'camelscl', 'hysets', 'lamah']
    names = ['Shared\ndomains', 'CAMELS-US', 'CAMELS-BR', 'CAMELS-CL', 'HYSETS', 'LamaH']
    xx = np.arange(len(domains))
    means = np.array([prediction[d]['gain_mean'] for d in domains])
    ci = np.array([prediction[d]['bootstrap_95_ci']['gain_mean'] for d in domains])
    predicted = np.array([prediction[d]['predicted_gain_mean'] for d in domains])
    ax.errorbar(xx-.09, means, yerr=np.array([means-ci[:,0],ci[:,1]-means]),
                fmt='o', ms=5.3, color=COLORS['Monthly donor'],
                capsize=3, elinewidth=1, capthick=1,
                label='Observed (95% CI)', zorder=3)
    ax.plot(xx+.09, predicted, linestyle='none', marker='D', ms=4.8,
            color=COLORS['Monthly blend'], label='Predicted', zorder=3)
    ax.axhline(0, color='#92979B', linewidth=.75)
    ax.set_xlim(-.55, 5.55)
    ax.set_ylim(-.34, .56)
    ax.set_yticks([-.2, 0, .2, .4])
    ax.set_xticks(xx, names)
    ax.set_ylabel('Annual-FDC error − monthly error')
    ax.legend(loc='upper left', ncol=2, columnspacing=1.4,
              handletextpad=.5, borderaxespad=.6)
    ax.yaxis.grid(True, color='#E3E5E6', linewidth=.5)
    ax.set_axisbelow(True)
    for d,m,c,p in zip(domains, means, ci, predicted):
        rows += [dict(figure='4',panel='b',setting=d,series='Observed',
                      estimate=m,low=c[0],high=c[1]),
                 dict(figure='4',panel='b',setting=d,series='Predicted',
                      estimate=p,low='',high='')]
    save(fig, args.out_dir, 'fig04_prediction_controls')

    # Paired differences keep the small blend effect visible on its own scale.
    fig, axes = plt.subplots(2, 1, figsize=(7.007874, 5.0))
    fig.subplots_adjust(left=.14, right=.98, top=.85, bottom=.09, hspace=.56)
    for ax, letter, field, title in zip(axes, ['a','b'],
            ['four_labels','discharge_only_bfi_flashiness'],
            ['Four process labels', 'Discharge-only labels (BFI and flashiness)']):
        label(ax, letter, title)
        for method, display, dx, marker in [
                ('annual_fdc','Annual-FDC blend',-.18,'s'),
                ('allocation','Allocation blend',0,'^'),
                ('monthly_w2','Monthly blend',.18,'o')]:
            vals, intervals = [], []
            for w in WINDOWS:
                r=control[w]['methods'][method][field]['versus_persistence']
                # Positive reduction favours the blend; intervals reverse too.
                v=-r['estimate']*1000
                lo,hi=-r['ci95'][1]*1000,-r['ci95'][0]*1000
                vals.append(v); intervals.append([v-lo,hi-v])
                rows.append(dict(figure='S2',panel=letter,setting=w,series=display,
                                 estimate=v,low=lo,high=hi))
            ax.errorbar(x+dx, vals, yerr=np.array(intervals).T,
                        color=COLORS[display], fmt=marker, ms=5,
                        elinewidth=1, capsize=2.6, capthick=1, label=display,
                        zorder=3)
        ax.axhline(0, color='#92979B', linewidth=.75)
        ax.set_xlim(-.58,3.58)
        ax.set_xticks(x, DISPLAY)
        ax.set_ylabel('Error reduction (0.001 units)')
        ax.yaxis.grid(True,color='#E3E5E6',linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(-3,7.5)
    axes[1].set_ylim(-4.7,4)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.57,.99),ncol=3,
                   columnspacing=1.4,handletextpad=.5)
    save(fig,args.out_dir,'figS02_blend_paired_controls')
    with (args.out_dir/'prediction_controls_source_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['figure','panel','setting','series','estimate','low','high'])
        writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    main()
