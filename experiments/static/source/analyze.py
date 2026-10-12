#!/usr/bin/env python3
"""Read an I16 result snapshot and write descriptive, world-clustered analyses.

Only Python's standard library is required. No significance threshold or research
continuation decision is encoded here. Probabilities and oracle values are on [0,1].
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import tempfile


GROUP_FIELDS = ("model", "stage", "condition", "attribution", "template_id", "split")
MATCH_FIELDS = ("world_id", "template_id", "prior", "reliability", "signal", "answer", "split")
REQUIRED_META = ("world_id", "stage", "condition", "attribution", "template_id", "oracle", "split")


def probability(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_manifest(path):
    if path is None:
        return {}, None
    raw = Path(path).read_bytes()
    source = raw.decode("utf-8-sig")
    try:
        content = json.loads(source)
        if isinstance(content, dict):
            content = content.get("items", content.get("rows", content))
        if isinstance(content, dict) and "id" in content:
            content = [content]
    except json.JSONDecodeError:
        content = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not isinstance(content, list):
        raise ValueError("Manifest must be a JSON list, an items/rows object, or JSONL.")
    items = {}
    for row in content:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("Every manifest row needs an id.")
        item_id = str(row["id"])
        if item_id in items:
            raise ValueError(f"Duplicate manifest id: {item_id}")
        items[item_id] = row
    return items, hashlib.sha256(raw).hexdigest()


def read_snapshot(db_path, manifest):
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        rows = [dict(row) for row in connection.execute("SELECT * FROM results ORDER BY model, item_id")]
    finally:
        connection.close()
    output, issues = [], Counter()
    observed = defaultdict(set)
    for row in rows:
        model, item_id = str(row["model"]), str(row["item_id"])
        observed[model].add(item_id)
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
        except (ValueError, TypeError):
            metadata = {}
            issues["metadata_parse_errors"] += 1
        expected = manifest.get(item_id)
        # Never replace recorded experimental metadata with a later manifest.
        if expected is not None and any(metadata.get(k) != expected.get(k)
                                        for k in REQUIRED_META + ("prior", "reliability", "signal", "answer", "prompt")):
            issues["manifest_metadata_mismatch"] += 1
            row["analysis_exclusion"] = "manifest_metadata_mismatch"
        if any(k not in metadata for k in REQUIRED_META):
            issues["missing_required_metadata"] += 1
            row["analysis_exclusion"] = "missing_required_metadata"
        if not probability(metadata.get("oracle")):
            issues["invalid_oracle"] += 1
            row["analysis_exclusion"] = "invalid_oracle"
        if row.get("status") == "ok" and not probability(row.get("probability")):
            issues["invalid_ok_probability"] += 1
            row["analysis_exclusion"] = "invalid_ok_probability"
        row["meta"] = metadata
        row["model"] = model
        row["observed"] = True
        output.append(row)
    for model, completed in observed.items():
        for item_id in sorted(manifest.keys() - completed):
            output.append({"id": f"{model}:{item_id}", "model": model,
                           "item_id": item_id, "status": "unobserved",
                           "probability": None, "meta": manifest[item_id], "observed": False})
    return output, dict(issues), len(rows)


def valid(row):
    return (row.get("status") == "ok" and not row.get("analysis_exclusion")
            and probability(row.get("probability")) and probability(row["meta"].get("oracle")))


def percentile(ordered, q):
    location = (len(ordered) - 1) * q
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def estimate(observations, label, reps, seed):
    """Equal weight per world; correlated templates are averaged before resampling."""
    worlds = defaultdict(list)
    for world_id, value in observations:
        if math.isfinite(value):
            worlds[str(world_id)].append(float(value))
    values = [statistics.fmean(worlds[key]) for key in sorted(worlds)]
    result = {"estimate": statistics.fmean(values) if values else None,
              "n_observations": sum(map(len, worlds.values())), "n_worlds": len(values),
              "ci95": None, "ci_method": "percentile bootstrap of equally weighted world means"}
    if len(values) < 2 or reps < 1:
        result["ci_unavailable_reason"] = "fewer than two worlds" if len(values) < 2 else "bootstrap disabled"
        return result
    digest = hashlib.sha256((str(seed) + ":" + label).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    count = len(values)
    draws = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count))
                   for _ in range(reps))
    result["ci95"] = [percentile(draws, .025), percentile(draws, .975)]
    return result


def group_summaries(rows, reps, seed):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) if field == "model" else row["meta"].get(field)
                    for field in GROUP_FIELDS)
        groups[key].append(row)
    result = []
    for key in sorted(groups, key=canonical):
        members = groups[key]
        kept = [row for row in members if valid(row)]
        stats = dict(zip(GROUP_FIELDS, key))
        stats.update({"n_items": len(members), "n_observed": sum(row["observed"] for row in members),
                      "n_valid": len(kept), "status_counts": dict(Counter(row.get("status", "unknown") for row in members)),
                      "analysis_exclusion_counts": dict(Counter(row["analysis_exclusion"] for row in members if row.get("analysis_exclusion")))})
        for metric, transform in (("mae", abs), ("signed_residual", lambda x: x)):
            values = [(row["meta"]["world_id"], transform(row["probability"] - row["meta"]["oracle"])) for row in kept]
            stats[metric] = estimate(values, canonical(key) + metric, reps, seed)
        result.append(stats)
    return result


def paired_analysis(rows, name, cell_names, key_fields, classify, compute, reps, seed):
    """Strict matching; duplicated, missing, invalid or incompatible cells are not used."""
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cell = classify(row["meta"])
        if cell not in cell_names:
            continue
        key = tuple(row["meta"].get(field) for field in key_fields)
        cells[canonical(key)][cell].append(row)
    counts = Counter({"candidate_pairs_or_sets": len(cells), "complete_valid_pairs_or_sets": 0,
                      "missing_cell_sets": 0, "duplicate_cell_sets": 0,
                      "invalid_or_unobserved_sets": 0, "incompatible_metadata_sets": 0})
    missing_cells, invalid_cells = Counter(), Counter()
    measurements = defaultdict(list)
    template_measurements = defaultdict(lambda: defaultdict(list))
    for cell_map in cells.values():
        absent = [cell for cell in cell_names if cell not in cell_map]
        if absent:
            counts["missing_cell_sets"] += 1
            missing_cells.update(absent)
            continue
        if any(len(cell_map[cell]) != 1 for cell in cell_names):
            counts["duplicate_cell_sets"] += 1
            continue
        selected = {cell: cell_map[cell][0] for cell in cell_names}
        unusable = [cell for cell, row in selected.items() if not valid(row)]
        if unusable:
            counts["invalid_or_unobserved_sets"] += 1
            invalid_cells.update(unusable)
            continue
        values = compute(selected)
        if values is None:
            counts["incompatible_metadata_sets"] += 1
            continue
        counts["complete_valid_pairs_or_sets"] += 1
        first = selected[cell_names[0]]["meta"]
        for metric, value in values.items():
            measurements[metric].append((first["world_id"], value))
            template_measurements[str(first["template_id"])][metric].append((first["world_id"], value))
    return {"name": name, "counts": dict(counts), "missing_cells": dict(missing_cells),
            "invalid_or_unobserved_cells": dict(invalid_cells),
            "metrics": {key: estimate(values, name + key, reps, seed) for key, values in sorted(measurements.items())},
            "by_template": {template: {key: estimate(values, name + template + key, reps, seed)
                                        for key, values in sorted(metrics.items())}
                            for template, metrics in sorted(template_measurements.items())}}


def label_direction(meta):
    answer = meta.get("answer")
    return 2 * int(answer) - 1 if answer in (0, 1) else None


def attribution_values(cells):
    left, right = cells["self"], cells["other"]
    if not math.isclose(left["meta"]["oracle"], right["meta"]["oracle"], abs_tol=1e-12):
        return None
    direction = label_direction(left["meta"])
    if direction is None:
        return None
    delta = left["probability"] - right["probability"]
    return {"self_minus_other_probability": delta,
            "self_minus_other_directional_evidence_weight": delta * direction,
            "self_minus_other_absolute_error": abs(left["probability"] - left["meta"]["oracle"])
            - abs(right["probability"] - right["meta"]["oracle"])}


def dependence_values(cells):
    same, independent = cells["same"], cells["independent"]
    meta = same["meta"]
    direction = label_direction(meta)
    if direction is None or meta.get("signal") != meta.get("answer"):
        return None
    observed = (independent["probability"] - same["probability"]) * direction
    normative = (independent["meta"]["oracle"] - meta["oracle"]) * direction
    return {"directional_dependence_sensitivity": observed,
            "normative_directional_sensitivity": normative,
            "sensitivity_residual": observed - normative}


def interaction_values(cells):
    self_values = dependence_values({"same": cells["same_self"], "independent": cells["independent_self"]})
    other_values = dependence_values({"same": cells["same_other"], "independent": cells["independent_other"]})
    for condition in ("same", "independent"):
        if not math.isclose(cells[condition + "_self"]["meta"]["oracle"],
                            cells[condition + "_other"]["meta"]["oracle"], abs_tol=1e-12):
            return None
    if self_values is None or other_values is None:
        return None
    return {"self_minus_other_dependence_sensitivity": self_values["directional_dependence_sensitivity"]
            - other_values["directional_dependence_sensitivity"]}


def baseline_values(cells):
    target, baseline = cells["target"], cells["baseline"]
    meta = target["meta"]
    if meta.get("signal") != meta.get("answer"):
        return None
    direction = label_direction(meta)
    if direction is None:
        return None
    observed = (target["probability"] - baseline["probability"]) * direction
    normative = (meta["oracle"] - baseline["meta"]["oracle"]) * direction
    return {"directional_update_over_baseline": observed,
            "normative_directional_update": normative,
            "update_residual": observed - normative}


def core_analyses(rows, reps, seed):
    collections = defaultdict(list)
    for row in rows:
        if row["meta"].get("stage") == "core":
            collections[(row["model"], row["meta"].get("split"))].append(row)
    output = []
    for (model, split), members in sorted(collections.items(), key=lambda item: canonical(item[0])):
        analyses = []
        for condition in ("same", "independent"):
            selected = [row for row in members if row["meta"].get("condition") == condition]
            analyses.append(paired_analysis(selected, f"attribution_{condition}", ("self", "other"), MATCH_FIELDS,
                                            lambda m: m.get("attribution"), attribution_values, reps, seed))
        for attribution in ("self", "other"):
            selected = [row for row in members if row["meta"].get("attribution") == attribution
                        and row["meta"].get("signal") == row["meta"].get("answer")]
            analyses.append(paired_analysis(selected, f"dependence_{attribution}", ("same", "independent"), MATCH_FIELDS,
                                            lambda m: m.get("condition"), dependence_values, reps, seed))
        selected = [row for row in members if row["meta"].get("signal") == row["meta"].get("answer")]
        analyses.append(paired_analysis(selected, "attribution_by_dependence_interaction",
                                        ("same_self", "independent_self", "same_other", "independent_other"),
                                        MATCH_FIELDS, lambda m: str(m.get("condition")) + "_" + str(m.get("attribution")),
                                        interaction_values, reps, seed))
        baseline_keys = tuple(field for field in MATCH_FIELDS if field != "answer")
        for condition in ("same", "independent"):
            for attribution in ("self", "other"):
                selected = [row for row in members if row["meta"].get("condition") == "baseline"
                            or (row["meta"].get("condition") == condition
                                and row["meta"].get("attribution") == attribution
                                and row["meta"].get("signal") == row["meta"].get("answer"))]
                analyses.append(paired_analysis(selected, f"baseline_update_{condition}_{attribution}", ("target", "baseline"),
                                                baseline_keys, lambda m: "baseline" if m.get("condition") == "baseline" else "target",
                                                baseline_values, reps, seed))
        output.append({"model": model, "split": split, "analyses": analyses})
    return output


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def metric_text(metric):
    ci = metric.get("ci95")
    return fmt(metric.get("estimate")) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else " [CI unavailable]")


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(summary):
    lines = ["# I16 实验描述性分析", "", f"生成时间：{summary['generated_at']}", "",
             "本报告不执行显著性筛选或自动决定研究去留。预实验结果属于探索性证据；正式确认需事先冻结方案并使用独立数据。",
             "", "同一 world 内的证据取值与模板变体先取均值，再给各 world 相同权重；95% 区间为按 world 重采样的百分位 bootstrap。"
             "区间反映这里采样的世界之间的变化，不代表模型训练种子、所有自然语言表述或其他模型的总体不确定性。",
             "无效、截断、运行错误和未观测样本不进入数值分析，不补值；因此存在失败时结果针对有效且配对完整的样本。",
             "", "## 完成情况", "", "| 模型 | 记录数 | 有效 | 状态 | 分析排除 |", "|---|---:|---:|---|---|"]
    for model in summary["models"]:
        lines.append(f"| {md(model['model'])} | {model['n_observed']} | {model['n_valid']} | {md(canonical(model['status_counts']))} | {md(canonical(model['analysis_exclusion_counts']))} |")
    if not summary["models"]:
        lines.append("| 尚无模型结果 | 0 | 0 | — | — |")
    lines += ["", "元数据检查：`" + canonical(summary["data_quality_issues"]) + "`", "",
              "## 各条件误差", "", "| 模型 / split | 阶段 / 条件 / 归属 / 模板 | 有效 / 总数 | worlds | MAE [95% CI] | 有符号残差 [95% CI] |",
              "|---|---|---:|---:|---|---|"]
    for row in summary["groups"]:
        label = " / ".join(md(row[field]) for field in ("stage", "condition", "attribution", "template_id"))
        lines.append(f"| {md(row['model'])} / {md(row['split'])} | {label} | {row['n_valid']} / {row['n_items']} | {row['mae']['n_worlds']} | {metric_text(row['mae'])} | {metric_text(row['signed_residual'])} |")
    lines += ["", "## 完整配对对照", "",
              "方向化归属效应 = (p_self − p_other) × (2A − 1)，正值表示 self 标签增强了对旧答案方向的支持。",
              "依赖敏感性 = (p_independent − p_same) × (2A − 1)，只在 S=A 的合法配对世界中计算；"
              "同时报告对应规范差值及残差。四格交互比较 self 与 other 的依赖敏感性。",
              "对照基线的更新用于分别检查同源重复计数和独立证据的有效使用。所有差值都在相同模型、split、world 和模板内配对。", ""]
    for collection in summary["core"]:
        lines += [f"### {md(collection['model'])} / {md(collection['split'])}", "",
                  "| 对照 | 完整 / 候选 | 缺格 | 重复格 | 无效或未观测 | 不兼容 |", "|---|---:|---:|---:|---:|---:|"]
        for analysis in collection["analyses"]:
            count = analysis["counts"]
            lines.append(f"| {analysis['name']} | {count['complete_valid_pairs_or_sets']} / {count['candidate_pairs_or_sets']} | {count['missing_cell_sets']} | {count['duplicate_cell_sets']} | {count['invalid_or_unobserved_sets']} | {count['incompatible_metadata_sets']} |")
        lines += ["", "| 对照 / 指标 | 世界加权估计 [95% CI] | worlds | 配对单元数 |", "|---|---|---:|---:|"]
        for analysis in collection["analyses"]:
            for name, metric in analysis["metrics"].items():
                lines.append(f"| {analysis['name']} / {name} | {metric_text(metric)} | {metric['n_worlds']} | {metric['n_observations']} |")
        lines += ["", "各模板单独结果和每种缺失单元计数见 summary.json 的 by_template 与 missing_cells 字段。", ""]
    lines += ["## 解读边界", "",
              "- 先检查 calibration 的规则理解、输出有效率与模板差异；规则理解失败本身不能证明自答偏差。",
              "- self 和 other 都忽视来源依赖，主要支持一般依赖推理困难；需要归属的完整配对差值来判断标签是否进一步改变证据使用。",
              "- 一律忽略旧答案可能减少同源重复计数，却损害独立证据更新；必须联合查看两类条件及仅有 S 的基线。",
              "- 归属操纵测量提示中的 self/other 标签，不等价于识别隐藏的心理自我承诺或真实生成历史。",
              "- 接近零或区间跨零不能单独证明等价；这些探索性区间未作多重比较校正，也不用于追逐阳性结果。",
              "- 有限或缺失配对会影响精度与样本构成。未提供 manifest 时，尚未写入数据库的整组样本不可见。", ""]
    return "\n".join(lines)


def analyze(db, output_dir, manifest_path=None, reps=2000, seed=16006):
    manifest, manifest_hash = load_manifest(manifest_path)
    rows, issues, n_db_rows = read_snapshot(db, manifest)
    models = []
    for model in sorted({row["model"] for row in rows}):
        members = [row for row in rows if row["model"] == model]
        models.append({"model": model, "n_observed": sum(row["observed"] for row in members),
                       "n_valid": sum(valid(row) for row in members),
                       "status_counts": dict(Counter(row["status"] for row in members)),
                       "analysis_exclusion_counts": dict(Counter(row["analysis_exclusion"] for row in members if row.get("analysis_exclusion"))),
                       "manifest_items_per_model": len(manifest) if manifest_path else None})
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "database": str(Path(db).resolve()), "manifest": str(Path(manifest_path).resolve()) if manifest_path else None,
               "manifest_sha256": manifest_hash, "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "n_database_rows": n_db_rows, "bootstrap_reps": reps, "seed": seed,
               "estimand": "mean over world means; evidence values and templates within a world are correlated variants, not independent observations",
               "interpretation": "descriptive; no automated scientific go/no-go; pilot is exploratory",
               "data_quality_issues": issues, "models": models,
               "groups": group_summaries(rows, reps, seed), "core": core_analyses(rows, reps, seed)}
    output = Path(output_dir)
    # Both files are generated from the same read transaction and replaced atomically.
    report = render_report(summary)
    atomic_write(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_write(output / "report.md", report)
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
    summary = analyze(args.db, args.output_dir, args.manifest, args.bootstrap_reps, args.seed)
    print(json.dumps({"n_database_rows": summary["n_database_rows"],
                      "models": summary["models"], "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
