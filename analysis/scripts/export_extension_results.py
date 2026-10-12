"""Export terminal extension evidence to exact CSV and compact appendix tables.

No data are bundled and no result is inferred from an unfinished run. Examples:

  python scripts/export_extension_results.py --gate GATE.json --validate-only
  python scripts/export_extension_results.py --gate GATE.json \
      --history-stage1 PARENTS.json --history-stage2 CHILDREN.json --cohort COHORT.json \
      --native-analysis NATIVE.json --native-manifest NATIVE.jsonl --native-db NATIVE.sqlite \
      --frozen-source FROZEN_SOURCE_DIRECTORY

Repeat --gate, --history-stage1, --history-stage2, --cohort or --contrast-analysis
for separate model files. --contrast-analysis accepts any terminal extension
analysis to retain its complete contrasts in CSV without adding another table.

CSV values use round-trip float representations. Only LaTeX display is rounded.
Native configuration MAE is recomputed with the exact frozen snapshot/describe
functions after all input hashes match. It describes valid reports within each
configuration, separately by protocol. It is not the complete-16-cell estimand.
Inputs are read only. Existing different generated files require --overwrite.
--validate-only creates no files. No imports beyond the standard library are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    'analyze_extension.py': 'c80073264cee768051ecf7b3a2a501172ce1b5eae9d2c9e39f5c59369f57eef5',
    'build_extension.py': '506a4b7707f42fcf4bad755b8af6e56246e92cbf9a05237037abf3bdafb20437',
}
TERMINAL = ('ok', 'invalid', 'truncated', 'error')
ARMS = ('native_self', 'replay_self', 'replay_other', 'replay_neutral')
MODELS = {'qwen3-8b-thinking': 'Qwen3-8B', 'phi4-reasoning': 'Phi-4-reasoning',
          'mistral-nemo-instruct': 'Mistral-Nemo-Instruct-2407'}
GROUP_ORDER = ('positive', 'zero', 'negative', 'baseline', 'independent')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_describe(row, location):
    n, valid = row['n_planned'], row['n_valid']
    require(integer(n) and n > 0 and integer(valid) and valid <= n, location+': invalid denominator')
    counts = row['status_counts']
    require(set(counts) <= set(TERMINAL), location+': contains pending, duplicate, or nonterminal records')
    require(all(integer(v) for v in counts.values()) and sum(counts.values()) == n,
            location+': terminal counts do not sum to the planned denominator')
    require(valid <= counts.get('ok', 0), location+': valid count exceeds ok count')
    require(numeric(row['valid_rate']) and math.isclose(row['valid_rate'], valid/n, abs_tol=1e-14),
            location+': inconsistent valid rate')
    mae = row['mae']
    require(mae['n_observations'] == valid and ((valid == 0 and mae['estimate'] is None) or
            (valid > 0 and numeric(mae['estimate']) and 0 <= mae['estimate'] <= 1)),
            location+': invalid MAE estimate or denominator')


def load_analysis(path, module=None):
    doc = read_json(path)
    require(doc.get('analysis_kind') == 'i16-extension-v1' and doc.get('schema_version') == 1,
            'Expected frozen extension analysis: '+path.name)
    for key in ('manifest_sha256', 'database_sha256'):
        require(isinstance(doc.get(key), str) and re.fullmatch('[0-9a-f]{64}', doc[key]),
                path.name+': missing input hash '+key)
    require(isinstance(doc['models'], list) and doc['models'], path.name+': no model results')
    require(len({r['model'] for r in doc['models']}) == len(doc['models']), path.name+': duplicate model')
    for row in doc['models']:
        require(module is None or row['module'] == module, path.name+': unexpected module')
        check_describe(row['overall'], path.name+'/'+row['model'])
        conditions = row['conditions']
        require(len({canonical(c['stratum']) for c in conditions}) == len(conditions), 'Duplicate condition stratum')
        for condition in conditions:
            check_describe(condition, path.name+'/'+row['model']+'/condition')
        require(sum(c['n_planned'] for c in conditions) == row['overall']['n_planned'] and
                sum(c['n_valid'] for c in conditions) == row['overall']['n_valid'],
                path.name+': conditions do not partition the model denominator')
        combined = Counter()
        for condition in conditions:
            combined.update(condition['status_counts'])
        require(dict(combined) == row['overall']['status_counts'], path.name+': condition status totals disagree')
    return doc


def provenance(path, role, doc=None):
    entry = {'role': role, 'file': path.name, 'sha256': sha(path)}
    if doc:
        entry.update({k: doc[k] for k in ('manifest_sha256', 'database_sha256') if k in doc})
    return entry


def gate_rows(paths, sources):
    output, seen = [], set()
    for path in paths:
        raw = read_json(path)
        if isinstance(raw, list):
            gates = raw
            doc = None
        else:
            doc = load_analysis(path, 'gate')
            gates = [row['qualification'] for row in doc['models']]
        sources.append(provenance(path, 'qualification', doc))
        for gate in gates:
            model, scope = gate['model'], gate['qualification_scope']
            require((model, scope) not in seen, 'Duplicate qualification source for '+model)
            seen.add((model, scope))
            expected = {'independent': 32} if scope == 'independent_extension_only' else {
                'positive': 16, 'zero': 16, 'negative': 16, 'baseline': 16, 'independent': 32}
            require(scope in ('five_groups', 'independent_extension_only'), 'Unexpected qualification scope')
            require(gate['thresholds'] == {'minimum_valid_rate': .95, 'maximum_family_mae': .05},
                    'Qualification thresholds differ from the frozen protocol')
            groups = {g['group']: g for g in gate['groups']}
            require(set(groups) == set(expected) and len(groups) == len(gate['groups']), 'Incomplete gate groups')
            computed = []
            model_rows = []
            for name in GROUP_ORDER:
                if name not in groups:
                    continue
                group = groups[name]
                check_describe(group, model+'/'+name)
                require(group['n_planned'] == expected[name], 'Gate group denominator mismatch')
                mae = group['mae']['estimate']
                passes = group['n_valid']/group['n_planned'] >= .95 and numeric(mae) and mae <= .05 and not group['exclusions']
                require(group['passes'] is passes, 'Stored gate-group decision disagrees with the frozen rule')
                computed.append(passes)
                model_rows.append({'model': model, 'qualification_scope': scope, 'group': name,
                    'n_planned': group['n_planned'], 'n_valid': group['n_valid'], 'valid_rate': group['valid_rate'],
                    'mae': mae, 'n_families': group['mae']['n_families'], 'group_passes': passes,
                    'model_passes': gate['passes'], **{status: group['status_counts'].get(status, 0) for status in TERMINAL},
                    'exclusions_json': canonical(group['exclusions']), 'family_mae_json': canonical(group['mae']['family_means']),
                    'source_file': path.name, 'source_sha256': sha(path)})
            require(type(gate['passes']) is bool and gate['passes'] == all(computed),
                    'Stored model gate decision disagrees with its groups')
            output.extend(model_rows)
    return output


def index_models(paths, module, sources):
    result = {}
    for path in paths:
        document = load_analysis(path, module)
        sources.append(provenance(path, module, document))
        for row in document['models']:
            require(row['model'] not in result, 'Duplicate '+module+' source for '+row['model'])
            result[row['model']] = (row, document, path)
    return result


def summed_counts(conditions):
    counts = Counter()
    for condition in conditions:
        counts.update(condition['status_counts'])
    n = sum(c['n_planned'] for c in conditions)
    valid = sum(c['n_valid'] for c in conditions)
    return {'n_planned': n, 'n_valid': valid, 'valid_rate': valid/n if n else None,
            **{status: counts.get(status, 0) for status in TERMINAL}}


def history_rows(args, sources):
    if not (args.history_stage1 or args.history_stage2 or args.cohort):
        return [], [], []
    require(args.history_stage1 and args.cohort, 'History export requires parent analyses and cohort ledgers')
    parents = index_models(args.history_stage1, 'history_stage1', sources)
    children = index_models(args.history_stage2, 'history_stage2', sources)
    cohorts = {}
    for path in args.cohort:
        ledger = read_json(path)
        entries = ledger['parents']
        require(len({(r['model'], r['parent_item_id']) for r in entries}) == len(entries), 'Duplicate cohort parent')
        require(all(type(r['eligible']) is bool for r in entries), 'Invalid parent eligibility')
        require(ledger['planned_parents'] == len(entries) and ledger['eligible_parents'] == sum(r['eligible'] for r in entries)
                and ledger['planned_children'] == 16*ledger['eligible_parents'], 'Invalid cohort totals')
        for model in sorted({r['model'] for r in entries}):
            require(model not in cohorts, 'Duplicate cohort ledger for '+model)
            cohorts[model] = ([r for r in entries if r['model'] == model], ledger, path)
        sources.append(provenance(path, 'history_cohort'))
    require(set(cohorts) == set(parents) and set(children) <= set(parents), 'History models do not match across inputs')
    parent_rows, arm_rows, evidence_rows = [], [], []
    for model, (parent, parent_doc, parent_path) in parents.items():
        entries, cohort, cohort_path = cohorts[model]
        require(cohort['stage1_manifest_sha256'] == parent_doc['manifest_sha256'] and
                cohort['stage1_db_sha256'] == parent_doc['database_sha256'], 'Cohort ledger identifies a different parent snapshot')
        require(len(entries) == parent['overall']['n_planned'] == 48, 'Expected 48 planned parents per history model')
        require(dict(Counter(r['status'] for r in entries)) == parent['overall']['status_counts'], 'Parent status counts disagree')
        eligible = sum(r['eligible'] for r in entries)
        require(eligible <= parent['overall']['n_valid'], 'Eligible cohort exceeds valid parents')
        parent_rows.append({'model': model, 'n_planned': 48, 'n_valid': parent['overall']['n_valid'],
            'n_eligible': eligible, 'n_ineligible': 48-eligible, 'planned_children': 16*eligible,
            'valid_rate': parent['overall']['valid_rate'], **{s: parent['overall']['status_counts'].get(s, 0) for s in TERMINAL},
            'source_file': parent_path.name, 'source_sha256': sha(parent_path),
            'cohort_file': cohort_path.name, 'cohort_sha256': sha(cohort_path)})
        if not eligible:
            require(model not in children, 'A zero-parent cohort cannot have child results')
            continue
        require(model in children, 'Eligible parents exist but terminal child analysis is missing for '+model)
        child, child_doc, child_path = children[model]
        require(child['overall']['n_planned'] == 16*eligible, 'Expected 16 formed children per eligible parent')
        require({c['stratum']['history_arm'] for c in child['conditions']} == set(ARMS), 'Missing or unexpected history arm')
        for arm in ARMS:
            selected = [c for c in child['conditions'] if c['stratum']['history_arm'] == arm]
            counts = summed_counts(selected)
            require(counts['n_planned'] == 4*eligible, 'Each arm must contain four branches per eligible parent')
            common = {'model': model, 'history_arm': arm, 'planned_parents': 48, 'eligible_parents': eligible,
                      'source_file': child_path.name, 'source_sha256': sha(child_path)}
            arm_rows.append({**common, **counts})
            require({c['stratum']['protocol'] for c in selected} == {'baseline', 'positive', 'independent'},
                    'Unexpected evidence protocols in history conditions')
            for protocol, evidence, multiplier in (('baseline', 'none', 1), ('positive', 'copy', 1), ('independent', 'independent', 2)):
                values = summed_counts([c for c in selected if c['stratum']['protocol'] == protocol])
                require(values['n_planned'] == multiplier*eligible, 'Incorrect evidence-branch denominator')
                evidence_rows.append({**common, 'evidence_kind': evidence, **values})
    return parent_rows, arm_rows, evidence_rows


def frozen_analyzer(folder):
    require(folder is not None, 'Native configuration MAE requires --frozen-source')
    for name, expected in FROZEN_HASHES.items():
        require((folder/name).is_file() and sha(folder/name) == expected, 'Frozen source hash mismatch: '+name)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        helper_spec = importlib.util.spec_from_file_location('build_extension', folder/'build_extension.py')
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        previous_helper = sys.modules.get('build_extension')
        sys.modules['build_extension'] = helper
        try:
            spec = importlib.util.spec_from_file_location('_frozen_i16_analyzer', folder/'analyze_extension.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if previous_helper is None:
                del sys.modules['build_extension']
            else:
                sys.modules['build_extension'] = previous_helper
    finally:
        sys.dont_write_bytecode = previous_bytecode
    return module


def native_rows(args, sources):
    if not args.native_analysis:
        require(not (args.native_manifest or args.native_db), 'Native raw inputs require --native-analysis')
        return [], []
    require(args.native_manifest and args.native_db, 'Native configuration MAE requires matching --native-manifest and --native-db')
    document = load_analysis(args.native_analysis, 'native_factorial')
    require(sha(args.native_manifest) == document['manifest_sha256'] and sha(args.native_db) == document['database_sha256'],
            'Native manifest or database hash differs from the terminal analysis')
    analyzer = frozen_analyzer(args.frozen_source)
    require(len(document['models']) == 1, 'The frozen native design contains one checkpoint')
    model = document['models'][0]
    require(model['model'] == 'qwen3-8b-thinking' and model['overall']['n_planned'] == 2304,
            'Native design differs from the frozen Qwen3 factorial design')
    manifest = analyzer.read_jsonl(args.native_manifest)
    records = analyzer.snapshot(manifest, args.native_db, model['model'])
    require(analyzer.describe(records) == model['overall'], 'Frozen describe does not exactly reproduce the source overall statistics')
    groups = defaultdict(list)
    for record in records:
        item = record['item']
        require(item['budget'] == item['max_new_tokens'], 'Budget differs from the actual configured cap')
        groups[(item['native_mode'], item['output_instruction'], item['budget'], item['protocol'])].append(record)
    expected = set(itertools.product(('on', 'off'), ('compatible', 'restrictive'), (4096, 12288), ('positive', 'independent')))
    require(set(groups) == expected, 'Incomplete native configuration grid')
    rows = []
    for mode, wording, budget, protocol in sorted(groups):
        detail = analyzer.describe(groups[(mode, wording, budget, protocol)])
        check_describe(detail, 'native configuration')
        require(detail['n_planned'] == (96 if protocol == 'positive' else 192), 'Native configuration denominator mismatch')
        rows.append({'model': model['model'], 'native_mode': mode, 'output_instruction': wording,
            'budget': budget, 'protocol': protocol, 'n_planned': detail['n_planned'], 'n_valid': detail['n_valid'],
            'valid_rate': detail['valid_rate'], 'mae': detail['mae']['estimate'], 'n_families': detail['mae']['n_families'],
            **{s: detail['status_counts'].get(s, 0) for s in TERMINAL}, 'exclusions_json': canonical(detail['exclusions']),
            'family_mae_json': canonical(detail['mae']['family_means']),
            'estimand': 'Equal family mean absolute error among valid reports in this configuration and protocol',
            'source_file': args.native_analysis.name, 'source_sha256': sha(args.native_analysis)})
    sources.extend([provenance(args.native_analysis, 'native_factorial', document),
                    provenance(args.native_manifest, 'native_manifest'), provenance(args.native_db, 'native_database'),
                    *[provenance(args.frozen_source/name, 'frozen_source') for name in FROZEN_HASHES]])
    conditions = []
    for condition in model['conditions']:
        conditions.append({'model': model['model'], **condition['stratum'], 'stratum_json': canonical(condition['stratum']), 'n_planned': condition['n_planned'],
            'n_valid': condition['n_valid'], 'valid_rate': condition['valid_rate'], 'mae': condition['mae']['estimate'],
            'detail_json': canonical({k: v for k, v in condition.items() if k != 'stratum'})})
    return rows, conditions


def contrast_rows(paths, sources):
    rows, seen = [], set()
    for path in paths:
        digest = sha(path)
        if digest in seen:
            continue
        seen.add(digest)
        doc = load_analysis(path)
        sources.append(provenance(path, 'complete_contrast_export', doc))
        for model in doc['models']:
            candidates = []
            for section in ('factorial', 'attribution', 'information'):
                for entry in model.get(section, []):
                    candidates.append((section, entry))
            for section in ('presentation', 'information'):
                for entry in model.get('history', {}).get(section, []):
                    candidates.append(('history_'+section, entry))
            for section, entry in candidates:
                summaries = entry.get('contrasts', {})
                summaries = {**summaries, **{k: v for k, v in entry.items()
                              if isinstance(v, dict) and 'candidate_sets' in v and 'point' in v}}
                for name, summary in summaries.items():
                    n, complete = summary['candidate_sets'], summary['complete_valid_sets']
                    require(integer(n) and integer(complete) and complete <= n and summary['point']['n_observations'] == complete,
                            'Invalid contrast denominator')
                    require((complete == 0 and summary['point']['estimate'] is None) or
                            (complete > 0 and numeric(summary['point']['estimate'])), 'Invalid contrast point')
                    bounds = summary['bounds']
                    require(bounds and bounds['n_planned_units'] == n and numeric(bounds['lower']) and
                            numeric(bounds['upper']) and bounds['lower'] <= bounds['upper']+1e-12,
                            'Invalid contrast bounds or planned denominator')
                    rows.append({'module': model['module'], 'model': model['model'], 'section': section,
                        'kind': entry.get('kind'), 'contrast': name, 'stratum_json': canonical(entry['stratum']),
                        'metric': summary['metric'], 'candidate_sets': n, 'complete_valid_sets': complete,
                        'point': summary['point']['estimate'], 'point_n_families': summary['point']['n_families'],
                        'bound_lower': bounds['lower'], 'bound_upper': bounds['upper'],
                        'bound_n_families': bounds['n_families'], 'n_unique_records': bounds['n_unique_records'],
                        'n_missing_records': bounds['n_missing_records'],
                        'normative_same_sets': summary.get('normative_same_sets', {}).get('estimate'),
                        'residual': summary.get('residual', {}).get('estimate'),
                        'summary_json': canonical(summary), 'source_file': path.name, 'source_sha256': digest})
    return rows


def csv_text(rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def tex(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(value))


def display(value, digits):
    if value is None:
        return 'n/a'
    if value != 0 and abs(value) < 10**(-digits):
        mantissa, exponent = f'{value:.1e}'.split('e')
        return '$'+mantissa+r'\!\times\!10^{'+str(int(exponent))+'}$'
    return f'{value:.{digits}f}'


def table_start(columns, caption, label, heading):
    return [r'\begin{table*}[t]', r'\centering', r'\begin{tabular}{@{}'+columns+'@{}}',
            r'\toprule', heading+r' \\', r'\midrule']


def table_end(caption, label, note=None):
    lines = [r'\bottomrule', r'\end{tabular}', r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{table*}']
    if note:
        lines += [note]
    return lines


def latex_tables(gates, parents, arms, native, digits):
    lines = ['% Generated only from terminal real evidence. Do not hand-edit this file.',
             '% CSV retains unrounded values and provenance records source hashes.', '']
    if gates:
        caption = 'Qualification results under the frozen validity and accuracy thresholds.'
        label = 'tab:extension-qualification'
        lines += table_start('llrrcc', caption, label, 'Model & Group & Valid/planned & MAE & Group pass & Model pass')
        previous = None
        for row in gates:
            first = row['model'] != previous
            if previous is not None and first:
                lines.append(r'\midrule')
            lines.append(' & '.join([tex(MODELS.get(row['model'], row['model'])) if first else '', tex(row['group']),
                f"{row['n_valid']}/{row['n_planned']}", display(row['mae'], digits),
                'Yes' if row['group_passes'] else 'No', ('Yes' if row['model_passes'] else 'No') if first else ''])+r' \\')
            previous = row['model']
        note = 'MAE gives equal weight to numerical families among valid reports. '
        if any(r['qualification_scope'] == 'independent_extension_only' for r in gates):
            note += 'The independent extension row supplements that checkpoint\'s existing qualification. '
        note += 'All planned gate records have terminal outcomes.'
        lines += table_end(caption, label, note)+['']
    if parents:
        caption = 'Coverage of initial generation and the four continuation arms.'
        label = 'tab:extension-history-coverage'
        lines += table_start('llrr', caption, label, 'Model & Stage or arm & Valid/planned & Eligible parents')
        for index, row in enumerate(parents):
            if index:
                lines.append(r'\midrule')
            lines.append(' & '.join([tex(MODELS.get(row['model'], row['model'])), 'Initial generation',
                f"{row['n_valid']}/{row['n_planned']}", f"{row['n_eligible']}/{row['n_planned']}"])+r' \\')
            for arm in arms:
                if arm['model'] == row['model']:
                    lines.append(' & '.join(['', tex(arm['history_arm'].replace('_', ' ')),
                        f"{arm['n_valid']}/{arm['n_planned']}", ''])+r' \\')
        note = ('Each eligible parent forms four evidence branches in each arm. Parent failures create no child prompts. '
                'Child denominators cover the formed cohort, and all reported planned records are terminal.')
        lines += table_end(caption, label, note)+['']
    # Native configuration CSVs are rendered by build_main_results.py in the main paper.
    if native:
        lines.append(r'Table~\ref{tab:extension-native-configurations} in the main paper gives all native configurations. Actual generated length is not used to select or group observations.')
    return '\n'.join(lines).rstrip()+'\n'


def save_outputs(outputs, overwrite):
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            require(overwrite, 'Different generated output already exists. Review it or use --overwrite: '+str(path))
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gate', type=Path, action='append', default=[])
    parser.add_argument('--history-stage1', type=Path, action='append', default=[])
    parser.add_argument('--history-stage2', type=Path, action='append', default=[])
    parser.add_argument('--cohort', type=Path, action='append', default=[])
    parser.add_argument('--native-analysis', type=Path)
    parser.add_argument('--native-manifest', type=Path)
    parser.add_argument('--native-db', type=Path)
    parser.add_argument('--frozen-source', type=Path)
    parser.add_argument('--contrast-analysis', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/extension')
    parser.add_argument('--tex-output', type=Path, default=ROOT/'sections/extension_results_generated.tex')
    parser.add_argument('--no-tex', action='store_true')
    parser.add_argument('--digits', type=int, default=4, help='LaTeX precision only; CSV remains unrounded')
    parser.add_argument('--validate-only', action='store_true', help='Read and validate actual inputs without writing any files')
    parser.add_argument('--check-source', action='store_true', help='Verify the frozen source hashes without reading or generating results')
    parser.add_argument('--overwrite', action='store_true', help='Replace only differing generated output paths after all inputs validate')
    args = parser.parse_args()
    require(1 <= args.digits <= 8, 'LaTeX digits must be between 1 and 8')
    if args.check_source:
        frozen_analyzer(args.frozen_source)
        print(json.dumps({'frozen_source_hashes_match': True, 'source_hashes': FROZEN_HASHES, 'result_files_created': 0}, indent=2))
        return
    require(args.gate or args.history_stage1 or args.history_stage2 or args.cohort or args.native_analysis or args.contrast_analysis,
            'Provide at least one actual evidence input')
    input_paths = [*args.gate, *args.history_stage1, *args.history_stage2, *args.cohort, *args.contrast_analysis]
    input_paths += [p for p in (args.native_analysis, args.native_manifest, args.native_db) if p]
    if args.native_analysis and args.frozen_source:
        input_paths += [args.frozen_source/name for name in FROZEN_HASHES]
    before = {p.resolve(): sha(p) for p in input_paths}
    sources = []
    gates = gate_rows(args.gate, sources)
    parents, arms, evidence = history_rows(args, sources)
    native, conditions = native_rows(args, sources)
    contrast_paths = [*args.history_stage2, *args.contrast_analysis]
    if args.native_analysis:
        contrast_paths.append(args.native_analysis)
    contrasts = contrast_rows(contrast_paths, sources)
    require(all(sha(p) == digest for p, digest in before.items()), 'An input changed during export')
    datasets = {'qualification_groups.csv': gates, 'history_parents.csv': parents,
                'history_arms.csv': arms, 'history_evidence_coverage.csv': evidence,
                'native_configurations.csv': native, 'native_condition_details.csv': conditions,
                'complete_contrasts.csv': contrasts}
    report = {'schema_version': 1, 'exporter_sha256': sha(Path(__file__)), 'sources': sources, 'inputs_unchanged': True,
        'rounding': {'csv': 'Unrounded round-trip floats; summary_json retains complete original contrast objects',
                     'latex_decimal_places': args.digits},
        'native_aggregation': {'keys': ['model', 'native_mode', 'output_instruction', 'budget', 'protocol'],
            'function': 'Frozen describe(snapshot(...)) grouped by the listed keys',
            'weighting': 'Equal family weights among valid records within each configuration and protocol',
            'estimand_distinction': 'Configuration description uses its own valid records. Factorial contrasts require complete sets of 16 cells.',
            'post_generation_grouping_used': False} if native else None,
        'history_count_aggregation': 'Sum disjoint frozen conditions within each history arm and protocol. No condition MAE means are averaged.',
        'rows': {name: len(rows) for name, rows in datasets.items() if rows}}
    if args.validate_only:
        print(json.dumps({'validation': 'passed', 'inputs_unchanged': True, 'rows': report['rows'], 'result_files_created': 0}, indent=2))
        return
    outputs = {args.output_dir/name: csv_text(rows).encode('utf-8') for name, rows in datasets.items() if rows}
    if not args.no_tex and (gates or parents or native):
        require(args.tex_output.resolve() not in {p.resolve() for p in outputs}, 'LaTeX output aliases a CSV output')
        outputs[args.tex_output] = latex_tables(gates, parents, arms, native, args.digits).encode('utf-8')
    require(not (set(p.resolve() for p in outputs) & set(before)), 'An output path aliases a read-only input')
    report['outputs'] = [{'file': path.name, 'sha256': hashlib.sha256(data).hexdigest()} for path, data in outputs.items()]
    ledger_path = args.output_dir/'export_provenance.json'
    require(ledger_path.resolve() not in before, 'Provenance output aliases a read-only input')
    require(ledger_path.resolve() not in {p.resolve() for p in outputs}, 'Provenance output aliases another output')
    outputs[ledger_path] = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    save_outputs(outputs, args.overwrite)
    print(json.dumps({'validation': 'passed', 'inputs_unchanged': True, 'rows': report['rows'],
                      'outputs': [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
