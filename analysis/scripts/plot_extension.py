"""Plot terminal I16 extension analyses without inference or output reparsing.

No default data, synthetic fixtures, or fallback estimates are supplied.
Run ``schema --analyzer PATH`` to check the frozen analyzer contract without
creating a figure. ``native`` and ``history`` require actual analysis JSON.
Use ``--validate-only`` on either plotting command to check its inputs only.
History additionally requires the stage-one analysis and frozen cohort ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
ANALYZER_SHA256 = 'c80073264cee768051ecf7b3a2a501172ce1b5eae9d2c9e39f5c59369f57eef5'
COLORS = {'self': '#2265C3', 'other': '#D99924', 'positive': '#258F96',
          'independent': '#2265C3', 'normative': '#68717D',
          'ink': '#101B36', 'grid': '#E6EBF0'}
MODELS = {'qwen3-8b-thinking': 'Qwen3-8B', 'phi4-reasoning': 'Phi-4-reasoning',
          'mistral-nemo-instruct': 'Mistral-Nemo-Instruct'}
TERMINAL = {'ok', 'invalid', 'truncated', 'error'}
MAIN_EFFECTS = ('native_mode', 'wording', 'budget')
EVIDENCE = ('none', 'copy', 'independent')
PRESENTATION = ('history_initial_choice_direction', 'replay_attribution_initial_choice_direction')
LEFT_STRATUM = {'protocol': 'independent', 'visibility': 'signal_visible', 'reminder': 'legacy',
                'output_instruction': 'restrictive', 'template_id': 'likelihood_v3', 'relation': 'agree'}
NEW_MODELS = ('phi4-reasoning', 'mistral-nemo-instruct')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def source(path):
    return {'file': path.name, 'sha256': sha(path)}


def exact_stratum(rows, wanted):
    matches = [row for row in rows if row.get('stratum') == wanted]
    require(len(matches) == 1, f'Expected exactly one pooled stratum {wanted}, found {len(matches)}')
    return matches[0]


def check_overall(row):
    overall = row['overall']
    planned, valid = overall['n_planned'], overall['n_valid']
    require(nonnegative_int(planned) and planned > 0 and nonnegative_int(valid) and valid <= planned,
            'Invalid planned/valid counts')
    counts = overall['status_counts']
    require(all(nonnegative_int(n) for n in counts.values()) and sum(counts.values()) == planned,
            'Status counts do not sum to the planned denominator')
    require(not (set(counts) - TERMINAL), 'The analysis contains pending, duplicate, or nonterminal records')
    require(valid <= counts.get('ok', 0), 'Valid count exceeds terminal ok records')
    return overall


def load_analysis(path, module, model=None):
    data = load_json(path)
    require(data.get('schema_version') == 1 and data.get('analysis_kind') == 'i16-extension-v1',
            'Expected frozen i16-extension-v1 analysis schema')
    for name in ('manifest_sha256', 'database_sha256'):
        require(isinstance(data.get(name), str) and re.fullmatch('[0-9a-f]{64}', data[name]),
                f'Missing or malformed {name}')
    candidates = [row for row in data['models'] if row['module'] == module and
                  (model is None or row['model'] == model)]
    require(len(candidates) == 1, 'Select exactly one model with --model')
    row = candidates[0]
    check_overall(row)
    return data, row


def check_summary(summary, metric, candidates, unique_records):
    require(summary['metric'] == metric, f'Expected {metric}, got {summary["metric"]}')
    require(summary['candidate_sets'] == candidates, 'Unexpected planned contrast denominator')
    n = summary['complete_valid_sets']
    require(nonnegative_int(n) and n <= candidates, 'Invalid complete-set count')
    point = summary['point']
    require(point['n_observations'] == n, 'Point estimate uses a different complete-set denominator')
    require((n == 0 and point['estimate'] is None) or (n > 0 and number(point['estimate'])),
            'Missing or nonfinite complete-set estimate')
    bounds = summary['bounds']
    require(isinstance(bounds, dict) and number(bounds['lower']) and number(bounds['upper']) and
            bounds['lower'] <= bounds['upper'] + 1e-12, 'Invalid finite-design bounds')
    require(bounds['n_planned_units'] == candidates and bounds['n_unique_records'] == unique_records,
            'Bound denominator or shared-record accounting mismatch')
    require(nonnegative_int(bounds['n_missing_records']) and bounds['n_missing_records'] <= unique_records,
            'Invalid missing-record denominator')
    # The complete-set point need not lie inside bounds for the full design.
    if metric == 'probability':
        oracle = summary['normative_same_sets']
        require(oracle['n_observations'] == n and
                ((n == 0 and oracle['estimate'] is None) or (n > 0 and number(oracle['estimate']))),
                'Normative point does not use the same complete sets')
    return summary


def native_payload(args):
    data, model = load_analysis(args.analysis, 'native_factorial', args.model)
    require(model['overall']['n_planned'] == 2304, 'Expected the frozen 2,304-call native factorial design')
    strata = {}
    for protocol, candidates in (('positive', 48), ('independent', 96)):
        row = exact_stratum(model['factorial'], {'protocol': protocol})
        selected = {}
        counts = set()
        for effect in MAIN_EFFECTS:
            for suffix, metric in (('mae', 'absolute_error'), ('attribution', 'probability')):
                key = effect + '_' + suffix
                selected[key] = check_summary(row['contrasts'][key], metric, candidates, 16*candidates)
                counts.add(selected[key]['complete_valid_sets'])
        require(len(counts) == 1, 'Native main contrasts must share the same complete 16-cell sets')
        strata[protocol] = selected
    conflict_row = exact_stratum(model['factorial'], {'protocol': 'independent', 'relation': 'conflict'})
    conflict = {suffix: check_summary(conflict_row['contrasts']['native_mode_'+suffix], metric, 48, 768)
                for suffix, metric in (('mae', 'absolute_error'), ('attribution', 'probability'))}
    return {'kind': 'native_factorial', 'model': model['model'], 'overall': model['overall'],
            'sources': {'analysis': source(args.analysis)},
            'manifest_sha256': data['manifest_sha256'], 'database_sha256': data['database_sha256'],
            'selection': 'Pooled templates, positive and independent kept separate. An additional row retains the prespecified independent conflict stratum. No relation/template rows are reaggregated.',
            'strata': strata, 'independent_conflict': conflict}


def history_payload(args):
    children_data, children = load_analysis(args.analysis, 'history_stage2', args.model)
    model = children['model']
    parent_data, parents = load_analysis(args.parents_analysis, 'history_stage1', model)
    cohort = load_json(args.cohort)
    require(cohort['stage1_manifest_sha256'] == parent_data['manifest_sha256'] and
            cohort['stage1_db_sha256'] == parent_data['database_sha256'],
            'Cohort ledger does not identify the supplied parent analysis snapshot')
    all_parents = cohort['parents']
    require(len({(r['model'], r['parent_item_id']) for r in all_parents}) == len(all_parents),
            'Duplicate parent in cohort ledger')
    require(all(type(r['eligible']) is bool for r in all_parents), 'Invalid cohort eligibility flag')
    require(cohort['planned_parents'] == len(all_parents) and
            cohort['eligible_parents'] == sum(r['eligible'] for r in all_parents) and
            cohort['planned_children'] == 16*cohort['eligible_parents'], 'Invalid global cohort accounting')
    cohort_rows = [r for r in all_parents if r['model'] == model]
    eligible = sum(r['eligible'] for r in cohort_rows)
    require(len(cohort_rows) == parents['overall']['n_planned'] == 48, 'Expected 48 planned parents per model')
    require(dict(Counter(r['status'] for r in cohort_rows)) == parents['overall']['status_counts'],
            'Parent status counts differ between cohort and parent analysis')
    require(0 < eligible <= parents['overall']['n_valid'],
            'No eligible parent cohort, or eligibility exceeds valid parents. Report parent coverage without a child figure.')
    require(children['overall']['n_planned'] == 16*eligible, 'Children denominator must be 16 per eligible parent')
    presentation = {}
    for evidence in EVIDENCE:
        n = eligible*(2 if evidence == 'independent' else 1)
        row = exact_stratum(children['history']['presentation'], {'evidence_kind': evidence})
        presentation[evidence] = {key: check_summary(row['contrasts'][key], 'probability', n, 2*n)
                                  for key in PRESENTATION}
    information = {}
    for evidence in ('copy', 'independent'):
        n = eligible*(2 if evidence == 'independent' else 1)
        row = exact_stratum(children['history']['information'],
                            {'evidence_kind': evidence, 'history_arm': 'native_self'})
        # Independent branches share the same no-new-evidence baseline.
        unique = eligible*(3 if evidence == 'independent' else 2)
        information[evidence] = check_summary(row['update'], 'probability', n, unique)
    result = {'kind': 'history_stage2', 'model': model, 'overall': children['overall'],
              'parents': {'planned': 48, 'valid': parents['overall']['n_valid'], 'eligible': eligible},
              'sources': {'analysis': source(args.analysis), 'parents_analysis': source(args.parents_analysis),
                          'cohort': source(args.cohort)},
              'manifest_sha256': children_data['manifest_sha256'],
              'database_sha256': children_data['database_sha256'],
              'selection': 'Pooled templates. Presentation contrasts use initial-choice direction. Native-self information contrasts use new-reading direction and the same-arm no-new-evidence baseline.',
              'presentation': presentation, 'information': information}
    if args.manifest:
        require(sha(args.manifest) == children_data['manifest_sha256'], 'Children manifest hash mismatch')
        rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        rows = [r for r in rows if not r.get('model_scope') or model in r['model_scope']]
        known = {r['parent_item_id']: r['parent_record_sha256'] for r in cohort_rows if r['eligible']}
        require(len(rows) == children['overall']['n_planned'] and all(
            r['parent_model'] == model and known.get(r['parent_item_id']) == r['parent_record_sha256'] for r in rows),
            'Children manifest is not linked to the supplied eligible parent cohort')
        require(Counter(r['parent_item_id'] for r in rows) == Counter({k: 16 for k in known}),
                'Children manifest does not contain 16 children per eligible parent')
        result['sources']['manifest'] = source(args.manifest)
    return result


def qualified_models(paths):
    records = {}
    for path in paths:
        document = load_json(path)
        if isinstance(document, list):
            rows = document
        else:
            require(document.get('analysis_kind') == 'i16-extension-v1', 'Unrecognized qualification document')
            rows = []
            for model in document['models']:
                require(model['module'] == 'gate', 'Qualification analysis must contain gate results')
                check_overall(model)
                rows.append(model['qualification'])
        for row in rows:
            model = row['model']
            require(model not in records, 'Duplicate qualification for '+model)
            require(type(row['passes']) is bool, 'Qualification result is not boolean')
            if row['qualification_scope'] != 'five_groups':
                continue
            require(model in NEW_MODELS, 'Unexpected new candidate in the frozen qualification plan')
            groups = {g['group']: g for g in row['groups']}
            expected = {'positive': 16, 'zero': 16, 'negative': 16, 'baseline': 16, 'independent': 32}
            require(set(groups) == set(expected) and len(row['groups']) == 5, 'Incomplete five-group qualification')
            require(row['thresholds'] == {'minimum_valid_rate': .95, 'maximum_family_mae': .05},
                    'Qualification thresholds differ from the frozen design')
            computed = []
            for name, planned in expected.items():
                group = groups[name]
                check_overall({'overall': group})
                require(group['n_planned'] == planned, 'Qualification group denominator mismatch')
                mae = group['mae']['estimate']
                passes = group['n_valid']/planned >= .95 and number(mae) and mae <= .05 and not group['exclusions']
                require(group['passes'] is passes, 'Stored gate decision disagrees with its actual group results')
                computed.append(passes)
            require(row['passes'] is all(computed), 'Stored qualification decision disagrees with all five groups')
            records[model] = row
    require(set(records) == set(NEW_MODELS), 'Provide terminal qualification results for both frozen new candidates')
    return records


def cross_model(model):
    require(model['overall']['n_planned'] == 1728, 'Expected the fixed 1,728-call cross-family subset')
    row = exact_stratum(model['attribution'], LEFT_STRATUM)
    left = {'self': check_summary(row['self_directional_error'], 'residual', 24, 24),
            'other': check_summary(row['other_directional_error'], 'residual', 24, 24),
            'gap': check_summary(row['gap'], 'probability', 24, 48)}
    require(len({r['complete_valid_sets'] for r in left.values()}) == 1,
            'Self, other and gap must use the same complete pairs')
    if left['gap']['complete_valid_sets']:
        require(math.isclose(left['self']['point']['estimate']-left['other']['point']['estimate'],
                             left['gap']['point']['estimate'], abs_tol=1e-12, rel_tol=0),
                'Attribution gap does not equal the same-pair self minus other directional errors')
    info_rows = [r for r in model['information'] if r['kind'] == 'answer_visibility_increment']
    right = {}
    for protocol, planned, unique in (('positive', 48, 96), ('independent', 96, 144)):
        row = exact_stratum(info_rows, {'protocol': protocol, 'attribution': 'self'})
        right[protocol] = check_summary(row['result'], 'probability', planned, unique)
    return {'model': model['model'], 'overall': model['overall'], 'decomposition': left, 'information': right}


def cross_payload(args):
    gates = qualified_models(args.qualification)
    data, reference = load_analysis(args.reference_analysis, 'cross_family', 'qwen3-8b-thinking')
    provenance = load_json(args.reference_provenance)
    require(provenance.get('reference_kind') == 'existing_r3_subset' and
            provenance.get('model') == reference['model'], 'Expected explicitly identified existing R3 reference')
    require(provenance.get('derived_manifest_sha256') == data['manifest_sha256'] and
            provenance.get('derived_database_sha256') == data['database_sha256'],
            'Reference mapping provenance does not match the analysis inputs')
    reference_result = cross_model(reference)
    reference_result['source_role'] = 'existing_r3_subset'
    reference_result['input_hashes'] = {key: data[key] for key in ('manifest_sha256', 'database_sha256')}
    models = {}
    sources = {'reference_analysis': source(args.reference_analysis),
               'reference_provenance': source(args.reference_provenance),
               'qualification': [source(path) for path in args.qualification], 'new_analysis': []}
    for path in args.analysis:
        document = load_json(path)
        require(document.get('analysis_kind') == 'i16-extension-v1', 'Expected extension analysis for a new candidate')
        for entry in document['models']:
            model = entry['model']
            require(entry['module'] == 'cross_family' and model in NEW_MODELS, 'Unexpected cross-family model or module')
            require(model not in models, 'Duplicate new cross-family result '+model)
            parsed, row = load_analysis(path, 'cross_family', model)
            require(gates[model]['passes'], 'Cross-family result is not qualified by the frozen five-group rule')
            models[model] = cross_model(row)
            models[model]['source_role'] = 'new_qualified_checkpoint'
            models[model]['input_hashes'] = {key: parsed[key] for key in ('manifest_sha256', 'database_sha256')}
        sources['new_analysis'].append(source(path))
    passed = {model for model, gate in gates.items() if gate['passes']}
    require(set(models) == passed, 'Every actual qualified candidate must be included, without selection by observed effects')
    return {'kind': 'cross_family', 'sources': sources,
            'models': [reference_result, *[models[m] for m in NEW_MODELS if m in models]],
            'left_stratum': LEFT_STRATUM,
            'right_stratum': {'visibility': 'signal_visible', 'attribution': 'self', 'reminder': 'scoped',
                              'output_instruction': 'compatible', 'templates': 'pooled', 'relation': 'pooled'},
            'qualification': {m: {**gates[m], 'scope': gates[m]['qualification_scope'],
                                  'excluded_from_cross_family': not gates[m]['passes'],
                                  'failed_groups': [g['group'] for g in gates[m]['groups'] if not g['passes']]}
                              for m in NEW_MODELS},
            'qualification_table_label': 'tab:extension-qualification',
            'reference_database_kind': provenance.get('database_kind'),
            'selection': 'The left condition is fixed before new results. Right updates use the frozen pooled-template output without reaggregation. R3 is an existing reference, not a new inference run.'}


def setup():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
        'axes.linewidth': .65, 'axes.edgecolor': COLORS['normative'], 'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'], 'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'axes.unicode_minus': True})


def clean(ax, values):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='x', length=3, width=.6)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.grid(axis='x', color=COLORS['grid'], linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    lo, hi = min([0., *values]), max([0., *values])
    span = max(hi-lo, .06)
    middle = (lo+hi)/2
    limits = (middle-.63*span, middle+.63*span)
    ax.set_xlim(*limits)
    ticks = MaxNLocator(nbins=3).tick_values(*limits)
    ax.set_xticks([value for value in ticks if limits[0] <= value <= limits[1]])
    ax.axvline(0, color=COLORS['normative'], lw=.65, ls='--', zorder=1)


def draw_estimate(ax, summary, y, color, counts=False, oracle=False):
    point, bounds = summary['point']['estimate'], summary['bounds']
    lower, upper = bounds['lower'], bounds['upper']
    bound_y = y-.095 if oracle else y-.07
    point_y = y+.12 if oracle else y+.07
    ax.plot([lower, upper], [bound_y, bound_y], color=color, lw=2., solid_capstyle='butt', zorder=2)
    ax.plot([lower, upper], [bound_y, bound_y], '|', color=color, ms=5, mew=.8, zorder=2)
    if point is not None:
        ax.plot(point, point_y, 'o', color=color, ms=4.7, zorder=4)
    values = [lower, upper] + ([] if point is None else [point])
    if oracle:
        norm = summary['normative_same_sets']['estimate']
        if norm is not None:
            ax.plot(norm, y, 'D', mfc='white', mec=COLORS['normative'], mew=.9, ms=4.8, zorder=4)
            values.append(norm)
    if counts:
        ax.text(1.02, y, f"{summary['complete_valid_sets']}/{summary['candidate_sets']}",
                transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=9.5)
    return values


def panel(fig, ax, title):
    box = ax.get_position()
    fig.text((box.x0+box.x1)/2, .024, title, ha='center', va='bottom', fontsize=10.5, fontweight='bold')


def native_figure(payload):
    fig = plt.figure(figsize=(6.3, 3.5))
    axes = [fig.add_axes([.215, .31, .34, .42]), fig.add_axes([.635, .31, .34, .42])]
    for ax, suffix in zip(axes, ('mae', 'attribution')):
        values = []
        for position, effect in zip((3, 2, 1), MAIN_EFFECTS):
            for protocol, offset in (('positive', .20), ('independent', -.20)):
                summary = payload['strata'][protocol][effect+'_'+suffix]
                values += draw_estimate(ax, summary, position+offset, COLORS[protocol])
        values += draw_estimate(ax, payload['independent_conflict'][suffix], 0, COLORS['independent'])
        ax.set(ylim=(-.52, 3.52), yticks=[3, 2, 1, 0])
        ax.axhline(.5, color=COLORS['grid'], linewidth=.6)
        clean(ax, values)
    axes[0].set_yticklabels(['On − off', 'Compatible\n− restrictive', '12,288 − 4,096', 'Conflict only:\non − off'])
    axes[1].set_yticklabels([])
    axes[0].set_xlabel('Change in MAE')
    axes[1].set_xlabel('Change in attribution gap')
    title = MODELS.get(payload['model'], payload['model']) + ' · native mode × wording × budget'
    fig.text(.5, .982, title, ha='center', va='top', fontsize=10.5)
    total = payload['overall']
    fig.text(.5, .913, f"Terminal outputs: {total['n_planned']:,}/{total['n_planned']:,} · valid: {total['n_valid']:,}/{total['n_planned']:,}",
             ha='center', va='top', fontsize=9.5)
    counts = [payload['strata'][p]['native_mode_mae'] for p in ('positive', 'independent')]
    fig.text(.5, .855, 'Complete sets of 16 cells: '+ ' · '.join(
        f"{p} {r['complete_valid_sets']}/{r['candidate_sets']}" for p, r in zip(('positive', 'independent'), counts)),
        ha='center', va='top', fontsize=9.5)
    handles = [Line2D([], [], color=COLORS[p], lw=2.5) for p in ('positive', 'independent')]
    handles += [Line2D([], [], color=COLORS['ink'], marker='o', lw=0, ms=4.7),
                Line2D([], [], color=COLORS['ink'], lw=2)]
    fig.legend(handles, ['Positive', 'Independent', 'Complete sets', 'Full design bounds'],
               loc='upper center', bbox_to_anchor=(.5, .812), frameon=False, ncol=4,
               columnspacing=1.1, handlelength=1.1, handletextpad=.35)
    conflict = payload['independent_conflict']['mae']
    fig.text(.5, .151, f"Conflict MAE bounds: [{conflict['bounds']['lower']:.4f}, {conflict['bounds']['upper']:.4f}]",
             ha='center', va='bottom', fontsize=9.5)
    fig.text(.5, .105, f"Conflict row: independent evidence, {conflict['complete_valid_sets']}/{conflict['candidate_sets']} complete blocks",
             ha='center', va='bottom', fontsize=9.5)
    panel(fig, axes[0], '(a) Accuracy contrasts')
    panel(fig, axes[1], '(b) Attribution contrasts')
    return fig


def history_figure(payload):
    fig = plt.figure(figsize=(6.3, 3.5))
    a = fig.add_axes([.19, .27, .30, .43])
    b = fig.add_axes([.735, .27, .20, .43])
    values = []
    for position, evidence in zip((2, 1, 0), EVIDENCE):
        for key, offset, color in zip(PRESENTATION, (.20, -.20), (COLORS['self'], COLORS['other'])):
            values += draw_estimate(a, payload['presentation'][evidence][key], position+offset, color, counts=True)
    a.set(ylim=(-.52, 2.52), yticks=[2, 1, 0], yticklabels=['No new\nevidence', 'Copy', 'Independent'],
          xlabel='Contrast toward initial choice')
    a.xaxis.label.set_fontsize(9.5)
    clean(a, values)
    extent = max(0.0001, max(abs(v) for v in values)*1.25)
    a.set_xlim(-extent, extent)
    a.xaxis.set_major_locator(MaxNLocator(nbins=3))
    a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value*1e4:g}'))
    a.set_xlabel('Contrast toward initial choice (×10⁻⁴)', fontsize=9.5)
    values = []
    for position, evidence in zip((1, 0), ('copy', 'independent')):
        values += draw_estimate(b, payload['information'][evidence], position, COLORS['self'], counts=True, oracle=True)
    b.set(ylim=(-.5, 1.5), yticks=[1, 0], yticklabels=['Copy', 'Independent'], xlabel='Directional update')
    clean(b, values)
    a.legend([Line2D([], [], marker='o', color=COLORS['self'], lw=0, ms=4.7),
              Line2D([], [], marker='o', color=COLORS['other'], lw=0, ms=4.7),
              Line2D([], [], color=COLORS['normative'], lw=2)],
             ['Native − replay self', 'Replay self − other', 'Full cohort bounds'], frameon=False,
             loc='lower left', bbox_to_anchor=(-.48, 1.015), borderaxespad=0, labelspacing=.20,
             handlelength=1.0, handletextpad=.4)
    b.legend([Line2D([], [], marker='o', color=COLORS['self'], lw=0, ms=4.7),
              Line2D([], [], marker='D', color=COLORS['normative'], mfc='white', lw=0, ms=4.8),
              Line2D([], [], color=COLORS['self'], lw=2)],
             ['Observed native self', 'Normative, same pairs', 'Full cohort bounds'], frameon=False,
             loc='lower left', bbox_to_anchor=(-.58, 1.015), borderaxespad=0, labelspacing=.20,
             handlelength=1.0, handletextpad=.4)
    fig.text(.5, .982, MODELS.get(payload['model'], payload['model'])+' · history of generated answers',
             ha='center', va='top', fontsize=10.5)
    parents, children = payload['parents'], payload['overall']
    parent_label = f"Parents: {parents['eligible']}/{parents['planned']} eligible"
    if parents['valid'] != parents['eligible']:
        parent_label += f" ({parents['valid']} valid)"
    fig.text(.5, .923, parent_label+f" · children: {children['n_valid']}/{children['n_planned']} valid, all terminal",
             ha='center', va='top', fontsize=9.5)
    fig.text(.5, .12, 'Counts: complete / planned pairs · bounds use the full eligible cohort',
             ha='center', va='bottom', fontsize=9.5)
    panel(fig, a, '(a) History and attribution')
    panel(fig, b, '(b) Native evidence updates')
    return fig


def cross_figure(payload):
    fig = plt.figure(figsize=(6.3, 3.5))
    a = fig.add_axes([.22, .27, .29, .46])
    b = fig.add_axes([.64, .27, .29, .46])
    models = payload['models']
    centers = [1.3*(len(models)-1-i) for i in range(len(models))]
    labels, left_values, right_values = [], [], []
    for row, center in zip(models, centers):
        model_name = MODELS.get(row['model'], row['model'])
        if row['source_role'] == 'existing_r3_subset':
            model_name += ' (R3)'
        model_name = model_name.replace('Mistral-Nemo-Instruct', 'Mistral-Nemo\nInstruct')
        n = row['decomposition']['gap']
        labels.append(model_name+f"\n{n['complete_valid_sets']}/{n['candidate_sets']} pairs")
        for key, offset, color in (('self', .33, COLORS['self']), ('other', 0., COLORS['other']),
                                   ('gap', -.33, COLORS['positive'])):
            left_values += draw_estimate(a, row['decomposition'][key], center+offset, color)
        for protocol, offset in (('positive', .30), ('independent', -.30)):
            right_values += draw_estimate(b, row['information'][protocol], center+offset,
                                          COLORS[protocol], counts=True, oracle=True)
    for ax, values in ((a, left_values), (b, right_values)):
        ax.set(ylim=(-.65, max(centers)+.65), yticks=centers)
        clean(ax, values)
        for center in centers[:-1]:
            ax.axhline(center-.65, color=COLORS['grid'], lw=.5)
    a.set_yticklabels(labels)
    b.set_yticklabels([])
    a.set_xlabel('Directional error or gap')
    b.set_xlabel('Directional update')
    title = ('Normative diagnosis in the Qwen3-8B R3 reference' if len(models) == 1
             else 'Normative diagnosis across checkpoints')
    fig.text(.5, .982, title, ha='center', va='top', fontsize=10.5)
    fig.text(.365, .916, 'Independent · agreement · likelihood_v3\nlegacy + restrictive',
             ha='center', va='top', fontsize=9.5, linespacing=1.1)
    fig.text(.785, .916, 'S visible · self attribution\nscoped + compatible · both templates',
             ha='center', va='top', fontsize=9.5, linespacing=1.1)
    a.legend([Line2D([], [], color=c, marker='o', lw=0, ms=4.7) for c in
              (COLORS['self'], COLORS['other'], COLORS['positive'])], ['Self', 'Other', 'Gap'],
             loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3, frameon=False,
             handlelength=.8, handletextpad=.25, columnspacing=.7, borderaxespad=0)
    b.legend([Line2D([], [], color=COLORS[p], marker='o', lw=0, ms=4.7) for p in ('positive', 'independent')],
             ['Copy', 'Independent'], loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=2,
             frameon=False, handlelength=.8, handletextpad=.25, columnspacing=.7, borderaxespad=0)
    fig.text(.5, .123, 'Dots: complete pairs · diamonds: oracle · bars: full design bounds',
             ha='center', va='bottom', fontsize=9.5)
    fig.text(.5, .075, 'Counts are complete / planned pairs. Bounds are not confidence intervals.',
             ha='center', va='bottom', fontsize=9.5)
    panel(fig, a, '(a) Attribution and normative error')
    panel(fig, b, '(b) Copy and independent evidence')
    return fig


def export(fig, payload, args):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text():
            require(text.get_fontsize() >= 9.5, 'Text smaller than 9.5 pt')
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                outside.append(text.get_text())
    require(not outside, 'Figure text exceeds the 6.3-inch canvas: '+repr(outside))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or {'native_factorial': 'extension_native_factorial',
                        'history_stage2': 'extension_history', 'cross_family': 'extension_cross_family'}[payload['kind']]
    require(re.fullmatch('[A-Za-z0-9_-]+', stem), 'Use a filename stem without paths')
    outputs = []
    for ext in ('pdf', 'svg'):
        target = args.output_dir/(stem+'.'+ext)
        fig.savefig(target)
        if ext == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in
                              target.read_text(encoding='utf-8').splitlines())+'\n',
                              encoding='utf-8', newline='\n')
        outputs.append(source(target))
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_dir/(stem+'.png'), dpi=200)
    payload['rendering'] = {'width_inches': float(fig.get_figwidth()), 'height_inches': float(fig.get_figheight()), 'font': 'Times New Roman',
                            'minimum_font_points': 9.5, 'text_within_canvas': True,
                            'point_and_bound_rows_separate': payload.get('point_and_bound_rows_separate', True),
                            'note': 'Bounds are deterministic missingness bounds, not confidence intervals.'}
    payload['outputs'] = outputs
    (args.output_dir/(stem+'.provenance.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    contract = commands.add_parser('schema', help='Check the frozen analyzer bytes; create no figures')
    contract.add_argument('--analyzer', type=Path, required=True)
    for name in ('native', 'history', 'cross'):
        p = commands.add_parser(name)
        if name == 'cross':
            p.add_argument('--analysis', type=Path, action='append', default=[], help='Terminal new cross-family analysis; repeat for separate files')
            p.add_argument('--qualification', type=Path, action='append', required=True, help='Actual terminal results for both new candidate gates; repeat as needed')
            p.add_argument('--reference-analysis', type=Path, required=True)
            p.add_argument('--reference-provenance', type=Path, required=True)
        else:
            p.add_argument('--analysis', type=Path, required=True)
            p.add_argument('--model')
        p.add_argument('--validate-only', action='store_true')
        p.add_argument('--output-dir', type=Path, default=ROOT/'figures')
        p.add_argument('--preview-dir', type=Path)
        p.add_argument('--stem')
        if name == 'history':
            p.add_argument('--parents-analysis', type=Path, required=True)
            p.add_argument('--cohort', type=Path, required=True)
            p.add_argument('--manifest', type=Path, help='Optionally verify each child links to its eligible parent record')
    args = parser.parse_args()
    if args.command == 'schema':
        require(sha(args.analyzer) == ANALYZER_SHA256, 'Analyzer differs from the frozen schema reviewed by this script')
        print(json.dumps({'schema_version': 1, 'analysis_kind': 'i16-extension-v1',
                          'analyzer_sha256': ANALYZER_SHA256, 'figures_created': 0}, indent=2))
        return
    payload = {'native': native_payload, 'history': history_payload, 'cross': cross_payload}[args.command](args)
    result = {'module': payload['kind'], 'schema_valid': True}
    if args.command == 'cross':
        result['models'] = [{'model': m['model'], 'role': m['source_role'],
                             'planned_records': m['overall']['n_planned'], 'valid_records': m['overall']['n_valid']}
                            for m in payload['models']]
    else:
        result.update(model=payload['model'], planned_records=payload['overall']['n_planned'],
                      valid_records=payload['overall']['n_valid'])
    if args.validate_only:
        result['figures_created'] = 0
    else:
        setup()
        figure = {'native': native_figure, 'history': history_figure, 'cross': cross_figure}[args.command](payload)
        result['outputs'] = export(figure, payload, args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
