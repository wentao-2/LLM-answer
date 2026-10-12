"""Plot all eight R3 independent-agreement conditions from retained CSV results.

Uses the original family-weighted estimates and bounds without reaggregation.
Run with --reference-analysis pointing to r3-qwen-cross-family-reference.json
in the delivered evidence archive (R4/analysis/).
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_extension import (ROOT, COLORS, setup, require, source, load_analysis,
                            cross_model, clean, draw_estimate, export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-analysis', type=Path, required=True)
    args = parser.parse_args()
    csv_path = ROOT/'data/full_results/attribution_decomposition.csv'
    rows = list(csv.DictReader(csv_path.open(encoding='utf-8-sig', newline='')))
    rows = [r for r in rows if all(r[k] == v for k, v in {
        'model': 'qwen3-8b-thinking', 'protocol': 'independent',
        'visibility': 'signal_visible', 'relation': 'agree'}.items())]
    require(len(rows) == 8, 'Expected all eight reminder/wording/template conditions')
    rows.sort(key=lambda r: (r['output_instruction'] != 'restrictive',
                            r['reminder'] != 'legacy', r['template_id']))
    _, reference = load_analysis(args.reference_analysis, 'cross_family', 'qwen3-8b-thinking')
    info = cross_model(reference)['information']
    setup()
    fig = plt.figure(figsize=(6.3, 4.2))
    a = fig.add_axes([.355, .28, .31, .55])
    b = fig.add_axes([.79, .35, .155, .35])
    values, labels, records = [], [], []
    for i, row in enumerate(rows):
        y = 7-i
        record = {k: row[k] for k in ('reminder', 'output_instruction', 'template_id', 'complete_pairs', 'candidate_pairs')}
        estimates = {}
        for metric, offset, color in (
            ('self_directional_error', .21, COLORS['self']),
            ('other_directional_error', 0, COLORS['other']),
            ('attribution_gap', -.21, COLORS['positive'])):
            value = float(row['metrics.'+metric+'.estimate'])
            lo = float(row['missing_probability_bounds.'+metric+'.lower'])
            hi = float(row['missing_probability_bounds.'+metric+'.upper'])
            a.plot([lo, hi], [y+offset, y+offset], color=color, lw=1.4, alpha=.7)
            a.plot(value, y+offset, 'o', color=color, ms=4)
            values.extend([value, lo, hi])
            estimates[metric] = {'estimate': value, 'lower': lo, 'upper': hi}
        require(abs(estimates['self_directional_error']['estimate']-estimates['other_directional_error']['estimate']-estimates['attribution_gap']['estimate']) < 1e-12, 'Same-pair identity failed')
        label = row['reminder'].capitalize()+' / '+row['output_instruction']+' / '+row['template_id'].split('_')[0]
        labels.append(label)
        a.text(1.01, y, row['complete_pairs']+'/24', transform=a.get_yaxis_transform(), fontsize=9.5, va='center')
        record['metrics'] = estimates
        records.append(record)
    a.set(ylim=(-.6, 7.6), yticks=list(range(7, -1, -1)), yticklabels=labels, xlabel='Directional error or gap')
    clean(a, values)
    a.axhline(3.5, color=COLORS['grid'], lw=1)
    a.set_xticks([-.2, -.1, 0, .05])
    a.text(1.015, 1.035, 'Pairs', transform=a.transAxes, fontsize=9.5)
    values = []
    for protocol, y in [('positive', 1), ('independent', 0)]:
        values += draw_estimate(b, info[protocol], y, COLORS[protocol], oracle=True)
    b.set(ylim=(-.65, 1.65), yticks=[1, 0], yticklabels=[], xlabel='Update')
    for y, label in [(1.37, 'Copy'), (.37, 'Independent')]:
        b.text(.5, y, label, transform=b.get_yaxis_transform(), ha='center', fontsize=9.5)
    clean(b, values)
    b.tick_params(axis='y', pad=2)
    fig.text(.5, .982, 'Qwen3-8B: conditional diagnosis and evidence updates', ha='center', va='top', fontsize=10.5)
    handles = [Line2D([], [], marker='o', color=c, lw=0, ms=4) for c in (COLORS['self'], COLORS['other'], COLORS['positive'])]
    fig.legend(handles, ['Self error', 'Other error', 'Attribution gap'], loc='upper center', bbox_to_anchor=(.40, .942), ncol=3, frameon=False, handletextpad=.3, columnspacing=.8)
    fig.text(.865, .83, 'Self attribution\nScoped / compatible\nBoth templates', ha='center', va='top', fontsize=9.5)
    fig.text(.5, .13, 'Dots: complete pairs. Bars: full design bounds. Diamonds: warranted updates.', ha='center', fontsize=9.5)
    counts = ', '.join(f"{name} {info[key]['complete_valid_sets']}/{info[key]['candidate_sets']}" for name, key in [('copy', 'positive'), ('independent', 'independent')])
    fig.text(.5, .087, 'Left: independent agreeing readings. Right: '+counts+' pairs.', ha='center', fontsize=9.5)
    fig.text(.39, .025, '(a) Attribution across eight conditions', ha='center', fontsize=10.5, fontweight='bold')
    fig.text(.85, .025, '(b) Evidence updates', ha='center', fontsize=10.5, fontweight='bold')
    payload = {'kind': 'attribution_diagnosis', 'point_and_bound_rows_separate': False,
               'sources': {'conditions_csv': source(csv_path), 'reference_analysis': source(args.reference_analysis)},
               'conditions': records, 'information': info,
               'selection': 'All eight R3 independent, signal-visible, agreeing conditions. No reaggregation.'}
    export(fig, payload, SimpleNamespace(output_dir=ROOT/'figures', stem='extension_normative_diagnosis', preview_dir=ROOT/'build/qa'))


if __name__ == '__main__':
    main()
