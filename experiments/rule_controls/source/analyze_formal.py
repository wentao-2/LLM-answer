#!/usr/bin/env python3
"""Candidate reminder analysis; finite-design summaries, no automatic scientific gates.

Raw observations are counted once. The single shared baseline is referenced in
both reminder arms without creating records. Pairing uses world identity, while
bootstrap resampling uses numerical family identity. This module does not edit data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import analyze as base


ARMS = ("explicit", "minimal")
GROUP_FIELDS = ("stage", "condition", "reminder", "attribution", "template_id", "split")
MARGIN = .02


def estimate(values, label, reps, seed):
    result = base.estimate(values, label, reps, seed)
    result["n_families"] = result.pop("n_worlds")
    result["ci_method"] = "percentile bootstrap of equally weighted numerical-family means"
    if "ci_unavailable_reason" in result:
        result["ci_unavailable_reason"] = result["ci_unavailable_reason"].replace("worlds", "families")
    return result


def validate_rows(rows, manifest, issues):
    issues = Counter(issues)
    for row in rows:
        meta = row["meta"]
        expected_arm = meta.get("reminder") in (ARMS if meta.get("condition") in ("same", "independent") else ("shared",))
        if not meta.get("family_id") or not expected_arm:
            row["analysis_exclusion"] = "invalid_formal_metadata"
            if row["observed"]:
                issues["invalid_formal_metadata"] += 1
        expected = manifest.get(str(row["item_id"]))
        if expected is not None and any(meta.get(k) != expected.get(k) for k in ("family_id", "reminder")):
            row["analysis_exclusion"] = "formal_manifest_metadata_mismatch"
            if row["observed"]:
                issues["formal_manifest_metadata_mismatch"] += 1
    return dict(issues)


def duplicate_prompts(rows, manifest):
    items = manifest or {str(row["item_id"]): row["meta"] for row in rows if row["observed"]}
    by_prompt = defaultdict(list)
    for item_id, meta in items.items():
        if isinstance(meta.get("prompt"), str):
            by_prompt[meta["prompt"]].append(item_id)
    duplicates = [{"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "item_ids": sorted(ids)}
                  for prompt, ids in by_prompt.items() if len(ids) > 1]
    return {"scope": "actual distinct manifest items; shared analytical references are not items",
            "n_items_checked": sum(map(len, by_prompt.values())), "n_duplicate_prompt_groups": len(duplicates),
            "n_items_in_duplicate_groups": sum(len(g["item_ids"]) for g in duplicates),
            "groups": sorted(duplicates, key=lambda group: group["prompt_sha256"])}


def group_summaries(rows, reps, seed):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model"],) + tuple(row["meta"].get(k) for k in GROUP_FIELDS)
        grouped[key].append(row)
    output = []
    for key, members in sorted(grouped.items(), key=lambda x: base.canonical(x[0])):
        kept = [row for row in members if base.valid(row)]
        result = dict(zip(("model",) + GROUP_FIELDS, key))
        result.update(n_items=len(members), n_observed=sum(row["observed"] for row in members), n_valid=len(kept),
                      status_counts=dict(Counter(row["status"] for row in members)))
        for metric, transform in (("mae", abs), ("signed_residual", lambda x: x)):
            values = [(row["meta"]["family_id"], transform(row["probability"] - row["meta"]["oracle"])) for row in kept]
            result[metric] = estimate(values, base.canonical(key) + metric, reps, seed)
        output.append(result)
    return output


def aggregate(records, label, reps, seed):
    metrics = sorted({key for record in records for key in record["metrics"]})
    output = {"n_paired_units": len(records),
              "n_worlds": len({record["world"] for record in records}),
              "n_families": len({record["family"] for record in records}),
              "metrics": {name: estimate([(r["family"], r["metrics"][name]) for r in records if name in r["metrics"]],
                                          label + name, reps, seed) for name in metrics}}
    if "absolute_effect" in metrics:
        output["maximum_absolute_effect"] = max(r["metrics"]["absolute_effect"] for r in records)
    return output


def paired(rows, name, required, classify, compute, reps, seed, key_fields=base.MATCH_FIELDS):
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        category = classify(row["meta"])
        if category not in required:
            continue
        key = tuple(row["meta"].get(field) for field in key_fields)
        cells[base.canonical(key)][category].append(row)
    counts = Counter(candidate_sets=len(cells), complete_valid_sets=0, missing_cell_sets=0,
                     duplicate_cell_sets=0, invalid_or_unobserved_sets=0, incompatible_metadata_sets=0)
    missing, invalid, records = Counter(), Counter(), []
    for group in cells.values():
        absent = [key for key in required if key not in group]
        if absent:
            counts["missing_cell_sets"] += 1
            missing.update(absent)
            continue
        if any(len(group[key]) != 1 for key in required):
            counts["duplicate_cell_sets"] += 1
            continue
        selected = {key: group[key][0] for key in required}
        bad = [key for key, row in selected.items() if not base.valid(row)]
        if bad:
            counts["invalid_or_unobserved_sets"] += 1
            invalid.update(bad)
            continue
        if len({row["meta"]["family_id"] for row in selected.values()}) != 1:
            counts["incompatible_metadata_sets"] += 1
            continue
        values = compute(selected)
        if values is None:
            counts["incompatible_metadata_sets"] += 1
            continue
        counts["complete_valid_sets"] += 1
        meta = selected[required[0]]["meta"]
        records.append({"world": meta["world_id"], "family": meta["family_id"],
                        "template": str(meta["template_id"]), "metrics": values})
    output = {"name": name, "counts": dict(counts), "missing_cells": dict(missing),
              "invalid_or_unobserved_cells": dict(invalid), **aggregate(records, name, reps, seed)}
    output["by_template"] = {template: aggregate([r for r in records if r["template"] == template], name + template, reps, seed)
                             for template in sorted({r["template"] for r in records})}
    return output


def effect_metrics(delta, direction):
    absolute = abs(delta)
    return {"signed_directional_effect": delta * direction, "signed_probability_effect": delta,
            "absolute_effect": absolute, "fraction_absolute_gt_0_02": float(absolute > .02),
            "fraction_absolute_gt_0_05": float(absolute > .05)}


def two_cell_effect(cells, left="self", right="other"):
    a, b = cells[left], cells[right]
    direction = base.label_direction(a["meta"])
    if direction is None or not math.isclose(a["meta"]["oracle"], b["meta"]["oracle"], abs_tol=1e-12):
        return None
    return effect_metrics(a["probability"] - b["probability"], direction)


def attribution_reminder_effect(cells):
    first = two_cell_effect({"self": cells["explicit_self"], "other": cells["explicit_other"]})
    second = two_cell_effect({"self": cells["minimal_self"], "other": cells["minimal_other"]})
    if first is None or second is None:
        return None
    if not math.isclose(cells["explicit_self"]["meta"]["oracle"], cells["minimal_self"]["meta"]["oracle"], abs_tol=1e-12):
        return None
    delta = first["signed_probability_effect"] - second["signed_probability_effect"]
    return effect_metrics(delta, base.label_direction(cells["explicit_self"]["meta"]))


def baseline_update(cells):
    target, baseline = cells["target"], cells["baseline"]
    direction = base.label_direction(target["meta"])
    if direction is None:
        return None
    observed = (target["probability"] - baseline["probability"]) * direction
    normative = (target["meta"]["oracle"] - baseline["meta"]["oracle"]) * direction
    return {"directional_update_over_baseline": observed, "normative_directional_update": normative,
            "update_residual": observed - normative}


def same_sign(meta):
    return meta.get("signal") == meta.get("answer")


def core_for_arm(members, arm, label, reps, seed):
    # These are references to existing rows. No DB records or raw counts are added.
    target = [row for row in members if row["meta"].get("reminder") == arm]
    baseline = [row for row in members if row["meta"].get("condition") == "baseline"]
    analyses = []
    for condition in ("same", "independent"):
        subsets = ("all",) if condition == "same" else ("all", "agree", "conflict")
        for subset in subsets:
            selected = [r for r in target if r["meta"].get("condition") == condition
                        and (subset == "all" or same_sign(r["meta"]) == (subset == "agree"))]
            name = f"attribution_{condition}_{subset}"
            analyses.append(paired(selected, label + name, ("self", "other"), lambda m: m.get("attribution"),
                                   two_cell_effect, reps, seed))
    agree = [row for row in target if same_sign(row["meta"])]
    for attribution in ("self", "other"):
        selected = [r for r in agree if r["meta"].get("attribution") == attribution]
        analyses.append(paired(selected, label + "dependence_" + attribution, ("same", "independent"),
                               lambda m: m.get("condition"), base.dependence_values, reps, seed))
    analyses.append(paired(agree, label + "attribution_by_dependence_interaction",
                           ("same_self", "independent_self", "same_other", "independent_other"),
                           lambda m: str(m.get("condition")) + "_" + str(m.get("attribution")),
                           base.interaction_values, reps, seed))
    baseline_keys = tuple(k for k in base.MATCH_FIELDS if k != "answer")
    for condition, subset in (("same", "agree"), ("independent", "agree"), ("independent", "conflict")):
        for attribution in ("self", "other"):
            selected = baseline + [r for r in target if r["meta"].get("condition") == condition
                                   and r["meta"].get("attribution") == attribution
                                   and same_sign(r["meta"]) == (subset == "agree")]
            analyses.append(paired(selected, label + f"baseline_update_{condition}_{subset}_{attribution}",
                                   ("target", "baseline"), lambda m: "baseline" if m.get("condition") == "baseline" else "target",
                                   baseline_update, reps, seed, baseline_keys))
    return {"reminder": arm, "shared_baseline_records_referenced": len(baseline), "analyses": analyses}


def core_analyses(rows, reps, seed):
    grouped = defaultdict(list)
    for row in rows:
        if row["meta"].get("stage") == "core":
            grouped[(row["model"], row["meta"].get("split"))].append(row)
    output = []
    for (model, split), members in sorted(grouped.items(), key=lambda item: base.canonical(item[0])):
        prefix = model + ":" + str(split) + ":"
        arms = [core_for_arm(members, arm, prefix + arm + ":", reps, seed) for arm in ARMS]
        reminders = []
        for condition in ("same", "independent"):
            for subset in (("all",) if condition == "same" else ("all", "agree", "conflict")):
                selected = [r for r in members if r["meta"].get("condition") == condition
                            and (subset == "all" or same_sign(r["meta"]) == (subset == "agree"))]
                name = prefix + f"reminder_effect_on_attribution_{condition}_{subset}"
                reminders.append(paired(selected, name, ("explicit_self", "explicit_other", "minimal_self", "minimal_other"),
                                        lambda m: str(m.get("reminder")) + "_" + str(m.get("attribution")),
                                        attribution_reminder_effect, reps, seed))
                for attribution in ("self", "other"):
                    attributed = [r for r in selected if r["meta"].get("attribution") == attribution]
                    reminders.append(paired(attributed, prefix + f"reminder_effect_{condition}_{subset}_{attribution}", ARMS,
                                            lambda m: m.get("reminder"),
                                            lambda cells: two_cell_effect(cells, "explicit", "minimal"), reps, seed))
        output.append({"model": model, "split": split, "arms": arms, "paired_reminder_comparisons": reminders})
    return output


def model_summaries(rows):
    models = []
    for model in sorted({row["model"] for row in rows}):
        members = [row for row in rows if row["model"] == model]
        kept = [row for row in members if base.valid(row)]
        probabilities = Counter(row["probability"] for row in kept)
        dominant = probabilities.most_common(1)
        models.append({"model": model, "n_items": len(members), "n_observed": sum(r["observed"] for r in members),
                       "n_valid": len(kept), "status_counts": dict(Counter(r["status"] for r in members)),
                       "analysis_exclusion_counts": dict(Counter(r["analysis_exclusion"] for r in members if r.get("analysis_exclusion"))),
                       "n_unique_valid_probabilities": len(probabilities), "dominant_probability": dominant[0][0] if dominant else None,
                       "dominant_probability_count": dominant[0][1] if dominant else 0,
                       "n_design_families": len({r["meta"].get("family_id") for r in members if r["meta"].get("family_id")})})
    return models


def reasoning_segment(raw_output):
    """Read emitted text only; an empty closed segment remains a valid observation."""
    raw = raw_output if isinstance(raw_output, str) else ""
    position = raw.find("</think>")
    if position < 0:
        return {"closed": False, "state": "unclosed", "nonempty": None,
                "reasoning_characters": None, "closing_tag_count": 0}
    prefix = raw[:position]
    if "<think>" in prefix:
        prefix = prefix.replace("<think>", "", 1)
    prefix = prefix.strip()
    nonempty = bool(prefix)
    return {"closed": True, "state": "nonempty" if nonempty else "empty", "nonempty": nonempty,
            "reasoning_characters": len(prefix), "closing_tag_count": raw.count("</think>")}


def thinking_enabled(row):
    meta = row["meta"]
    if isinstance(meta.get("thinking_enabled"), bool):
        return meta["thinking_enabled"]
    mode = str(meta.get("model_mode", "")).lower()
    if mode.startswith("nonthinking") or mode.startswith("instruct-direct"):
        return False
    if mode.startswith("thinking"):
        return True
    # Legacy rows may lack mode metadata; report the presence of native tags.
    raw = row.get("raw_output") or ""
    return True if "<think>" in raw or "</think>" in raw else None


def token_stats(rows, field="output_tokens", metadata=False):
    values = [(row["meta"] if metadata else row).get(field) for row in rows]
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v) and v >= 0]
    return {"n_available": len(values), "total": sum(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None}


def readout_diagnostics(rows, reps, seed):
    """Post-generation strata are descriptive, never used to exclude primary data."""
    grouped, coverage = defaultdict(list), Counter()
    for row in rows:
        if not row["observed"]:
            continue
        enabled = thinking_enabled(row)
        coverage[{True: "enabled_records", False: "disabled_records", None: "unknown_mode_records"}[enabled]] += 1
        if enabled is not True:
            continue
        key = (row["model"],) + tuple(row["meta"].get(k) for k in GROUP_FIELDS)
        grouped[key].append((row, reasoning_segment(row.get("raw_output"))))
    groups = []
    for key, members in sorted(grouped.items(), key=lambda pair: base.canonical(pair[0])):
        raw_rows = [row for row, _ in members]
        states = Counter(segment["state"] for _, segment in members)
        closed = states["empty"] + states["nonempty"]
        result = dict(zip(("model",) + GROUP_FIELDS, key))
        result.update(n_observed=len(members), n_closed=closed, n_empty=states["empty"],
                      n_nonempty=states["nonempty"], n_unclosed=states["unclosed"],
                      closed_rate=closed / len(members), empty_rate=states["empty"] / len(members),
                      nonempty_rate=states["nonempty"] / len(members),
                      empty_rate_among_closed=states["empty"] / closed if closed else None,
                      multiple_closing_tag_records=sum(segment["closing_tag_count"] > 1 for _, segment in members),
                      status_counts=dict(Counter(row["status"] for row in raw_rows)),
                      finish_reason_counts=dict(Counter(str(row["meta"].get("finish_reason", "unknown")) for row in raw_rows)),
                      output_tokens=token_stats(raw_rows),
                      provided_reasoning_tokens=token_stats(raw_rows, "reasoning_tokens", metadata=True))
        result["reasoning_nonempty_metadata_mismatches"] = sum(
            isinstance(row["meta"].get("reasoning_nonempty"), bool) and segment["closed"]
            and row["meta"]["reasoning_nonempty"] != segment["nonempty"] for row, segment in members)
        result["by_reasoning_state"] = {}
        for state in ("empty", "nonempty", "unclosed"):
            selected = [row for row, segment in members if segment["state"] == state]
            kept = [row for row in selected if base.valid(row)]
            values = [(row["meta"]["family_id"], abs(row["probability"] - row["meta"]["oracle"])) for row in kept]
            result["by_reasoning_state"][state] = {
                "n_observed": len(selected), "n_valid_for_mae": len(kept),
                "output_tokens": token_stats(selected),
                "mae": estimate(values, base.canonical(key) + state + "readout_mae", reps, seed)}
        groups.append(result)
    return {"scope": "all observed rows of thinking-enabled modes; native-tag presence is a fallback for legacy modes",
            "coverage": dict(coverage), "classification": "prefix before first </think>, optional <think> removed, whitespace stripped",
            "interpretation": "descriptive post-generation strata only; no filtering of primary metrics and no mediation or causal inference",
            "reasoning_token_policy": "use saved reasoning_tokens when available; do not infer tokenizer counts from characters",
            "groups": groups}


def all_comparisons(collection):
    for arm in collection["arms"]:
        yield from arm["analyses"]
    yield from collection["paired_reminder_comparisons"]


def render_report(summary):
    lines = ["# I16 规范提醒对照：有限设计分析", "", "生成时间：" + summary["generated_at"], "",
             "实际记录只计一次。baseline/calibration 只调用一次；同一个 baseline 在两种提醒条件中作为分析参照，未复制为新样本。",
             "配对使用 world、模板和证据取值；区间按 numerical family 聚类，互补先验、语境、符号和模板不增加独立 family 数。",
             "这些是指定模型和固定网格条件下的描述性估计。bootstrap 是对 family 构成的敏感性描述，不构成对自然语言或模型总体的抽样推断。",
             "", "预定义平均方向效应参照范围为 ±0.02。程序不自动宣告等价；常数输出或校准失败可产生虚假的零差异，"
             "必须联合检查概率判断与独立证据使用。均值接近零不排除抵消后的局部大差异。", "",
             "## 原始记录与测量诊断", "", "| 模型 | 实际记录 / 计划 | 有效 | 唯一概率数 | 最常见概率 / 次数 | family数 |",
             "|---|---:|---:|---:|---|---:|"]
    for m in summary["models"]:
        lines.append(f"| {base.md(m['model'])} | {m['n_observed']} / {m['n_items']} | {m['n_valid']} | {m['n_unique_valid_probabilities']} | {base.fmt(m['dominant_probability'])} / {m['dominant_probability_count']} | {m['n_design_families']} |")
    lines += ["", "元数据问题：`" + base.canonical(summary["data_quality_issues"]) + "`。",
              f"实际 manifest 中完全相同的提示组：{summary['exact_prompt_duplicates']['n_duplicate_prompt_groups']}；共享分析参照不计入此项。",
              "无效、截断、错误和未观测样本不补值；各状态及排除数见 summary.json，缺失条件会改变有效配对样本构成。", "",
              "## Thinking-enabled 输出诊断", "",
              "从原始输出首个 </think> 之前的文本判定闭合及空/非空；去除可选 <think> 标签和首尾空白。"
              "所有主分析样本保持不变。空/非空是生成后的描述性分层，其 MAE 差异不用于推断因果或中介作用。",
              "", "| 模型 | 阶段 / 依赖 / 提醒 / 归属 / 模板 | 已观测 | 闭合 / 空 / 非空 | 输出token均值 | 空 / 非空 MAE |",
              "|---|---|---:|---|---:|---|"]
    for g in summary["readout_diagnostics"]["groups"]:
        label = " / ".join(base.md(g[k]) for k in GROUP_FIELDS if k != "split")
        states = g["by_reasoning_state"]
        lines.append(f"| {base.md(g['model'])} | {label} | {g['n_observed']} | {g['n_closed']} / {g['n_empty']} / {g['n_nonempty']} | {base.fmt(g['output_tokens']['mean'])} | {base.fmt(states['empty']['mae']['estimate'])} / {base.fmt(states['nonempty']['mae']['estimate'])} |")
    lines += ["", "各比例、状态分层的token统计和有效MAE样本数见 summary.json；未来记录中的 reasoning_tokens 仅在实际保存时汇总，不从字数推测token数。", "",
              "## 规范误差", "", "| 模型 | 阶段 / 依赖 / 提醒 / 归属 / 模板 | 有效 / 计划 | MAE [95% 描述区间] | family数 |",
              "|---|---|---:|---|---:|"]
    for g in summary["groups"]:
        label = " / ".join(base.md(g[k]) for k in GROUP_FIELDS if k != "split")
        lines.append(f"| {base.md(g['model'])} | {label} | {g['n_valid']} / {g['n_items']} | {base.metric_text(g['mae'])} | {g['mae']['n_families']} |")
    lines += ["", "## 配对归属与提醒效应", "",
              "方向化归属效应为 (p_self−p_other)×(2A−1)；提醒效应采用 explicit−minimal。"
              "reminder_effect_on_attribution 是两种提醒下归属效应之差。最大值为所有有效配对的最大绝对变化。", "",
              "| 对照 | 完整 / 候选 | 方向效应 [95% 描述区间] | 平均绝对变化 | >2pp比例 / >5pp比例 | 最大绝对变化 | family数 |",
              "|---|---:|---|---:|---|---:|---:|"]
    for collection in summary["core"]:
        for a in all_comparisons(collection):
            metrics = a["metrics"]
            if "signed_directional_effect" not in metrics:
                continue
            lines.append(f"| {base.md(a['name'])} | {a['counts']['complete_valid_sets']} / {a['counts']['candidate_sets']} | {base.metric_text(metrics['signed_directional_effect'])} | {base.fmt(metrics['absolute_effect']['estimate'])} | {base.fmt(metrics['fraction_absolute_gt_0_02']['estimate'])} / {base.fmt(metrics['fraction_absolute_gt_0_05']['estimate'])} | {base.fmt(a.get('maximum_absolute_effect'))} | {a['n_families']} |")
    lines += ["", "## 依赖敏感性、交互和基线更新", "",
              "依赖差只在 S=A 的共同合法世界中计算。独立冲突证据另与仅 S 基线匹配，更新方向取旧答案 A 的方向。",
              "", "| 对照 / 指标 | 估计 [95% 描述区间] | 完整 / 候选 | family数 |", "|---|---|---:|---:|"]
    for collection in summary["core"]:
        for a in all_comparisons(collection):
            if "signed_directional_effect" in a["metrics"]:
                continue
            for key, metric in a["metrics"].items():
                lines.append(f"| {base.md(a['name'])} / {key} | {base.metric_text(metric)} | {a['counts']['complete_valid_sets']} / {a['counts']['candidate_sets']} | {metric['n_families']} |")
    lines += ["", "逐模板结果、无效/缺格/重复格计数和每个指标的 family 数见 summary.json。"
              "所有区间均为描述性区间，未作多重比较校正；完整有限网格上的实际均值本身没有抽样误差，"
              "这里的 bootstrap 仅展示对设计 family 构成的敏感性。", ""]
    return "\n".join(lines)


def analyze(db, output_dir, manifest_path=None, reps=2000, seed=16006):
    manifest, manifest_hash = base.load_manifest(manifest_path)
    rows, issues, count = base.read_snapshot(db, manifest)
    issues = validate_rows(rows, manifest, issues)
    summary = {"schema_version": 1, "analysis_kind": "candidate-reminder-family-analysis",
               "generated_at": datetime.now(timezone.utc).isoformat(), "n_database_rows": count,
               "database": str(Path(db).resolve()), "manifest_sha256": manifest_hash,
               "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "base_analysis_source_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
               "bootstrap_reps": reps, "seed": seed, "mean_directional_margin": MARGIN,
               "margin_interpretation": "prespecified design-conditional reference for mean directional effects only; no automatic equivalence conclusion",
               "estimand": "equal numerical-family means; full finite-grid means with descriptive composition-sensitivity bootstrap",
               "shared_baseline_policy": "one raw record; referenced separately by reminder analyses; never counted as a new record",
               "data_quality_issues": issues, "exact_prompt_duplicates": duplicate_prompts(rows, manifest),
               "models": model_summaries(rows), "groups": group_summaries(rows, reps, seed),
               "core": core_analyses(rows, reps, seed), "readout_diagnostics": readout_diagnostics(rows, reps, seed)}
    output = Path(output_dir)
    base.atomic_write(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    base.atomic_write(output / "report.md", render_report(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=16006)
    args = parser.parse_args()
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    result = analyze(args.db, args.output_dir, args.manifest, args.bootstrap_reps, args.seed)
    print(json.dumps({"n_database_rows": result["n_database_rows"], "models": result["models"],
                      "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
