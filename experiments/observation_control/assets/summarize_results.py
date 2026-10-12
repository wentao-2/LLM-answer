"""Descriptive summaries of frozen results, including all missing-output bounds."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from common import atomic_json, canonical, load, now, sha, validate_seal


def metric(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    values = {key: mean(row[field] for row in group if row.get(field) is not None)
              for key, group in groups.items() if any(row.get(field) is not None for row in group)}
    return dict(n=sum(row.get(field) is not None for row in rows),
                n_families=len(values), expected_families=len(groups), family_values=values,
                equal_family_mean=mean(values.values()) if values and len(values) == len(groups) else None,
                family_range=[min(values.values()), max(values.values())] if values else None)


def summarize(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return [dict(zip(keys, key), n_planned=len(group),
                 status_counts=dict(Counter(row['status'] for row in group if 'status' in row)),
                 metrics={field: metric(group, field) for field in fields})
            for key, group in sorted(groups.items(), key=lambda pair: canonical(pair[0]))]


def value(row, key):
    return row['metrics'][key]['equal_family_mean']


def number(x):
    return '不可估计' if x is None else f'{x:.6f}'


def main():
    validate_seal()
    result = load(ROOT / 'analysis/RESULTS.json')
    digest = sha(ROOT / 'analysis/RESULTS.json')
    for name in ('AUDIT', 'LOCAL_AUDIT'):
        audit = load(ROOT / 'audit' / (name + '.json'))
        assert audit['state'] == 'passed' and audit['analysis_sha256'] == digest
    errors = result['error_rows']
    assert len(errors) == 2880 and sum(sum(x.values()) for x in result['status_counts'].values()) == 2880
    transfer = [x for x in errors if x['stage'] == 'transfer']
    expression = [x for x in errors if x['stage'] == 'expression']
    pairs = [x for x in result['pair_rows'] if x['stage'] == 'transfer']
    contrasts = [x for x in result['contrast_rows'] if x['stage'] == 'transfer']
    assert len(transfer) == 2048 and len(expression) == 768
    em = ('posterior_error', 'error_lower', 'error_upper')
    pm = ('absolute_update_residual', 'absolute_update_lower', 'absolute_update_upper',
          'effective_weight', 'absolute_weight_residual')
    cm = ('posterior_error_change', 'error_change_lower', 'error_change_upper',
          'absolute_update_residual_change', 'absolute_update_change_lower', 'absolute_update_change_upper')
    report = dict(state='descriptive_summary', summarized_at=now(), analysis_sha256=digest,
                  design_seal_sha256=sha(ROOT / 'DESIGN_SEAL.json'), constructed_calls=2880,
                  status_counts=result['status_counts'],
                  transfer_conditions=summarize(transfer, ('model', 'condition'), em),
                  expression_conditions=summarize(expression, ('rule', 'condition'), em),
                  branch_updates=summarize(pairs, ('model', 'condition', 'rho', 'branch'), pm),
                  direction_context=summarize(transfer, ('model', 'condition', 'context', 'signal'), em),
                  paired_condition_contrasts=summarize(contrasts, ('model', 'comparison', 'condition', 'reference_condition'), cm),
                  work_schema_counts=[dict(model=model, n=len(rows),
                                            schema_valid=sum(x['work_schema_valid'] for x in rows),
                                            observed_correct=sum(x['observed_correct'] is True for x in rows),
                                            source_variables_correct=sum(x['not_revealed_names_correct'] is True for x in rows))
                                      for model in sorted(result['status_counts'])
                                      for rows in [[x for x in result['work_rows'] if x['model'] == model]]],
                  complete_parent_counts=[dict(model=model, stage=stage, n=n)
                                          for (model, stage), n in sorted(Counter(
                                              (x['model'], x['stage']) for x in result['complete_parent_ids']).items())],
                  manual_source_coding='pending',
                  interpretation_scope='fixed synthetic grid; descriptive family-weighted results, no hidden mechanism claim')
    atomic_json(ROOT / 'analysis/SUMMARY.json', report)
    names = {'ordinary': '通常提问', 'general_bayes': '提醒使用贝叶斯公式',
             'explicit': '明确使用全部已观察变量', 'full_steps': '列观测、来源和联合似然步骤',
             'general': '一般问题', 'direct': '自然语言规则', 'selector': '显式随机选择过程'}
    lines = [
        '# 新一轮实验完成情况与结果', '',
        '推理于北京时间 2026-10-12 04:25:31 完成。2,880 条调用全部保存，64 个新父答案全部有效。',
        '统计程序对辅助 WORK_JSON 的空输出处理曾报错，已通过单独包装程序恢复。冻结源码、最终概率解析规则和原始数据未变，原报错记录保留，未重跑推理。服务器独立审计、本地独立复核与完整快照 SHA 核验均通过。', '',
        '## 输出状态', '',
        '| 模型 | 全部调用 | 有效概率 | 格式不合规 | 长度截断 |',
        '|---|---:|---:|---:|---:|',
    ]
    for model, status in result['status_counts'].items():
        lines.append(f"| {model} | {sum(status.values())} | {status.get('ok', 0)} | {status.get('invalid', 0)} | {status.get('truncated', 0)} |")
    lines += ['', '## 新数值与两种上下文中的四种提问', '',
              '每模型每条件有 256 条计划续答。误差是最终概率与正确后验之差的绝对值，越小越好。以下按数值家庭等权汇总。有效输出均值仅描述可解析记录，完整分母界同时保留全部失败输出。没有把两种模型当作相同预算排名。', '',
              '| 模型 | 条件 | 有效/计划 | 有效输出平均误差 | 完整分母误差下界 | 完整分母误差上界 |',
              '|---|---|---:|---:|---:|---:|']
    for row in report['transfer_conditions']:
        lines.append(f"| {row['model']} | {names[row['condition']]} | {row['status_counts'].get('ok', 0)}/{row['n_planned']} | {number(value(row, 'posterior_error'))} | {number(value(row, 'error_lower'))} | {number(value(row, 'error_upper'))} |")
    lines += ['', 'Phi 的明确条件化提问保留全部有效输出，误差低于通常提问和一般贝叶斯提醒。Qwen 的有效输出呈同方向改善，但仍有截断，完整分母区间重叠。不能仅依据有效输出宣称 Qwen 在所有缺失情形下也稳定改善。',
              '完整步骤条件在 Phi 中有 99 条格式不合规，在 Qwen 中有 39 条截断。它不构成当前证据支持的稳定方案，失败结果没有被排除。', '',
              '## 等价规则表达与明确条件化对象', '',
              '24 个保存父答案全部有完整的 32 条阶段一续答，768 条均有效。每个下列条件有 192 条输出，所有条件都使用同一批父答案。', '',
              '| 等价规则表达 | 提问 | 平均绝对误差 |',
              '|---|---|---:|']
    for row in report['expression_conditions']:
        lines.append(f"| {names[row['rule']]} | {names[row['condition']]} | {number(value(row, 'posterior_error'))} |")
    lines += ['', '明确条件化要求在显式随机选择过程下改善明显，但在自然语言规则下整体误差略增。结果支持规则表达与条件化对象共同影响回答，不能把单一提示措辞描述为普遍有效，也不能由输出相似度推断模型的隐藏内部算法。', '',
              '## 完整分母的配对改进', '',
              '下表只列各干预相对通常提问的后验误差变化。负值表示改善。有效配对值使用双方都可解析的条目，完整分母下界和上界包含缺失条目。各数值家庭等权，各父记录重复分支不作为独立样本。', '',
              '| 模型 | 干预 | 有效配对/计划 | 配对误差变化 | 完整分母下界 | 完整分母上界 |',
              '|---|---|---:|---:|---:|---:|']
    for row in report['paired_condition_contrasts']:
        if row['comparison'] != 'intervention_minus_ordinary':
            continue
        m = row['metrics']['posterior_error_change']
        lines.append(f"| {row['model']} | {names[row['condition']]} | {m['n']}/{row['n_planned']} | {number(m['equal_family_mean'])} | {number(value(row, 'error_change_lower'))} | {number(value(row, 'error_change_upper'))} |")
    lines += ['', '## 证据入口与未完成项', '',
              '`analysis/RESULTS.json` 和八份 CSV 保留逐条概率、各条件自己的无新增证据基线、更新残差、证据权重、方向与上下文分组、配对对照及缺失界。`analysis/SUMMARY.json` 提供紧凑的条件与分支汇总，`assets/summarize_results.py` 可复算该说明。',
              '新数值阶段跨全部 32 条续答都有效的父记录为 Phi 4 个、Qwen 1 个。完整父子集很小，主要报告保留全计划分母和配对缺失界，不仅报告这个子集。',
              '来源说明的自由文本人工编码尚未完成。辅助结构化记录已有独立字段检查，但不能替代来源文字核对。本文尚未根据本轮结果改写，也未推送 Overleaf。', '',
              f"冻结设计 SHA256：`{report['design_seal_sha256']}`。", '',
              f"完整统计 SHA256：`{digest}`。", '']
    (ROOT / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    with (ROOT / 'analysis/condition_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('model', 'condition', 'planned', 'valid', 'valid_mae', 'mae_lower', 'mae_upper'))
        writer.writeheader()
        for row in report['transfer_conditions']:
            writer.writerow(dict(model=row['model'], condition=row['condition'], planned=row['n_planned'],
                                 valid=row['status_counts'].get('ok', 0), valid_mae=value(row, 'posterior_error'),
                                 mae_lower=value(row, 'error_lower'), mae_upper=value(row, 'error_upper')))
    print(canonical(dict(state='passed', constructed_calls=2880, analysis_sha256=digest,
                         summary_sha256=sha(ROOT / 'analysis/SUMMARY.json'), manual_source_coding='pending')))


if __name__ == '__main__':
    main()
