#!/usr/bin/env python3
"""Read-only finite-grid analysis for the frozen I16 mechanism follow-up.

No bootstrap, p-values, logit substitution, output repair, model selection by
attribution effects, or causal mediation inference is performed. A shared row
can be referenced by both analysis modules without becoming another observation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import tempfile


SCHEMA = ("id", "module", "analysis_modules", "protocol", "condition", "visibility",
          "attribution", "reminder", "output_instruction", "world_id", "family_id",
          "prior", "reliability", "signal", "answer", "oracle", "prompt",
          "template_id", "context_id")
GROUP_FIELDS = ("module", "protocol", "visibility", "attribution", "reminder",
                "output_instruction", "template_id")
WORLD_FIELDS = ("world_id", "context_id", "template_id", "prior", "reliability")
CELL_FIELDS = WORLD_FIELDS + ("protocol", "visibility", "signal", "answer",
                              "attribution", "reminder", "output_instruction")
PROTOCOLS = ("positive", "zero", "negative", "independent")
ATTRIBUTIONS = ("self", "other", "neutral")
REMINDERS = ("scoped", "legacy")
INSTRUCTIONS = ("compatible", "restrictive")
MAE_GATE = .05
VALID_GATE = .95


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix="." + path.name, delete=False) as stream:
            name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def compute_oracle(item):
    """Independent Bayes calculation; zero is an independent fair coin."""
    protocol, visibility = item["protocol"], item["visibility"]
    prior, reliability = item["prior"], item["reliability"]
    signal, answer = item["signal"], item["answer"]
    if visibility == "signal_hidden":
        observations = [] if protocol == "zero" else [answer if protocol == "positive" else 1 - answer]
    elif visibility in ("answer_hidden", "baseline"):
        observations = [signal]
    elif visibility == "signal_visible":
        observations = [signal, answer] if protocol == "independent" else [signal]
    else:
        raise ValueError("Unknown visibility")
    p1, p0 = prior, 1 - prior
    for observed in observations:
        p1 *= reliability if observed == 1 else 1 - reliability
        p0 *= reliability if observed == 0 else 1 - reliability
    return p1 / (p1 + p0)


def validate_item(item):
    missing = [key for key in SCHEMA if key not in item]
    if missing:
        raise ValueError(f"Missing manifest fields: {missing}")
    if item["module"] not in ("gate", "mechanism", "wording"):
        raise ValueError("Invalid physical module")
    memberships = item["analysis_modules"]
    if not isinstance(memberships, list) or len(set(memberships)) != len(memberships):
        raise ValueError("analysis_modules must be a unique list")
    if item["module"] not in memberships or any(x not in ("gate", "mechanism", "wording") for x in memberships):
        raise ValueError("Invalid analysis module membership")
    if item["protocol"] not in PROTOCOLS + ("baseline",) or item["condition"] != item["protocol"]:
        raise ValueError("Invalid protocol/condition")
    if item["reminder"] not in REMINDERS or item["output_instruction"] not in INSTRUCTIONS:
        raise ValueError("Invalid wording factors")
    if not probability(item["prior"]) or not 0 < item["prior"] < 1:
        raise ValueError("Invalid prior")
    if not probability(item["reliability"]) or not 0 < item["reliability"] < 1:
        raise ValueError("Invalid reliability")
    for key in ("id", "world_id", "family_id", "prompt", "context_id", "template_id"):
        if not isinstance(item[key], str) or not item[key]:
            raise ValueError(f"Invalid {key}")
    s, a, v, p = item["signal"], item["answer"], item["visibility"], item["protocol"]
    bit = lambda x: type(x) is int and x in (0, 1)
    if v == "signal_hidden":
        legal = s is None and bit(a) and p in ("positive", "zero", "negative")
    elif v == "signal_visible":
        legal = bit(s) and bit(a) and p in PROTOCOLS
        legal = legal and (p != "positive" or a == s) and (p != "negative" or a == 1 - s)
    elif v == "answer_hidden":
        legal = bit(s) and a is None and p in PROTOCOLS
    elif v == "baseline":
        legal = bit(s) and a is None and p == "baseline"
    else:
        legal = False
    if not legal:
        raise ValueError(f"Illegal protocol/visibility/evidence cell: {p}/{v}/{s}/{a}")
    if item["module"] == "gate":
        if item["attribution"] != "none" or memberships != ["gate"] or v not in ("signal_hidden", "baseline"):
            raise ValueError("Gate must be unattributed hidden calibration or baseline")
    elif item["attribution"] not in ATTRIBUTIONS or "gate" in memberships or p == "baseline":
        raise ValueError("Invalid main attribution/module")
    if "mechanism" in memberships and (item["reminder"] != "scoped" or item["output_instruction"] != "compatible"):
        raise ValueError("Mechanism cells require scoped/compatible")
    if "wording" in memberships and (v != "signal_visible" or p not in ("positive", "independent")
                                     or item["attribution"] not in ("self", "other")):
        raise ValueError("Invalid wording cell")
    if len(memberships) > 1 and set(memberships) != {"mechanism", "wording"}:
        raise ValueError("Only mechanism/wording sharing is allowed")
    if not probability(item["oracle"]) or not math.isclose(item["oracle"], compute_oracle(item), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Manifest oracle disagrees with the specified mechanism")


def load_manifest(path):
    raw = Path(path).read_bytes()
    content = raw.decode("utf-8-sig")
    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(items, dict):
        items = items.get("items", items.get("rows"))
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a nonempty item list or JSONL")
    result, cells = {}, set()
    for item in items:
        validate_item(item)
        if item["id"] in result:
            raise ValueError("Duplicate manifest id")
        cell = tuple(item[key] for key in CELL_FIELDS)
        if cell in cells:
            raise ValueError("Duplicate semantic cell: shared rows must be stored once")
        cells.add(cell)
        result[item["id"]] = item
    return result, hashlib.sha256(raw).hexdigest()


def read_snapshot(db_path, manifest, models=None):
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        stored = [dict(row) for row in db.execute("SELECT * FROM results ORDER BY model,item_id")]
    named = set(models or ())
    if not models:
        config_path = Path(db_path).parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            named.update(m["name"] for m in config.get("models", []))
    named.update(str(row["model"]) for row in stored)
    grouped, issues, details = defaultdict(list), Counter(), []
    for row in stored:
        grouped[(str(row["model"]), str(row["item_id"]))].append(row)
        if str(row["item_id"]) not in manifest:
            issues["unknown_manifest_item"] += 1
            details.append({"model": row["model"], "item_id": row["item_id"], "reason": "unknown_manifest_item"})
    planned = []
    for model in sorted(named):
        for item_id, expected in manifest.items():
            matches = grouped[(model, item_id)]
            row = {"model": model, "item_id": item_id, "meta": expected,
                   "observed": bool(matches), "status": "unobserved", "probability": None,
                   "raw_output": "", "recorded_metadata": {}, "analysis_exclusion": None}
            if len(matches) > 1:
                row.update(status="duplicate_records", analysis_exclusion="duplicate_model_item",
                           recorded_status_counts=dict(Counter(r["status"] for r in matches)))
                issues["duplicate_model_item"] += 1
            elif matches:
                row.update(matches[0])
                row["model"], row["item_id"], row["meta"] = model, item_id, expected
                try:
                    meta = json.loads(row.get("metadata_json") or "{}")
                    if not isinstance(meta, dict):
                        raise ValueError("Metadata is not an object")
                except (ValueError, TypeError):
                    meta = {}
                    row["analysis_exclusion"] = "metadata_parse_error"
                    issues["metadata_parse_error"] += 1
                row["recorded_metadata"] = meta
                mismatched = [key for key, value in expected.items() if key not in meta or meta[key] != value]
                if mismatched:
                    row["analysis_exclusion"] = "manifest_metadata_mismatch"
                    issues["manifest_metadata_mismatch"] += 1
                    row["mismatched_fields"] = mismatched
                if row["status"] not in ("ok", "invalid", "truncated", "error"):
                    row["analysis_exclusion"] = "unknown_output_status"
                    issues["unknown_output_status"] += 1
                if row["status"] == "ok" and not probability(row["probability"]):
                    row["analysis_exclusion"] = "invalid_ok_probability"
                    issues["invalid_ok_probability"] += 1
            if row["analysis_exclusion"]:
                details.append({"model": model, "item_id": item_id, "reason": row["analysis_exclusion"],
                                "mismatched_fields": row.get("mismatched_fields", [])})
            planned.append(row)
    return planned, dict(issues), details, stored


def valid(row):
    return row["status"] == "ok" and not row.get("analysis_exclusion") and probability(row.get("probability"))


def estimate(observations):
    families = defaultdict(list)
    for family, value in observations:
        if not math.isfinite(value):
            raise ValueError("Nonfinite analysis value")
        families[family].append(float(value))
    means = {family: statistics.fmean(values) for family, values in sorted(families.items())}
    return {"estimate": statistics.fmean(means.values()) if means else None,
            "n_observations": sum(map(len, families.values())), "n_families": len(means),
            "family_means": means, "ci95": None, "uncertainty_method": "none; fixed finite design, no sampling inference"}


def reasoning_state(row):
    if not row["observed"]:
        return "unobserved"
    meta, raw = row.get("recorded_metadata", {}), row.get("raw_output") or ""
    mode = str(meta.get("model_mode", "")).lower()
    enabled = meta.get("thinking_enabled")
    if enabled is False or mode.startswith(("nonthinking", "instruct-direct")):
        return "not_applicable"
    if enabled is not True and not mode.startswith("thinking") and "<think>" not in raw and "</think>" not in raw:
        return "not_measured"
    if "</think>" not in raw:
        return "unclosed"
    prefix = raw.split("</think>", 1)[0].replace("<think>", "", 1).strip()
    return "nonempty" if prefix else "empty"


def token_summary(rows, key="output_tokens", metadata=False):
    values = [(row.get("recorded_metadata", {}) if metadata else row).get(key) for row in rows if row["observed"]]
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0]
    return {"n_available": len(values), "total": sum(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None, "maximum": max(values) if values else None}


def describe(rows):
    kept = [row for row in rows if valid(row)]
    errors = [(row["meta"]["family_id"], abs(row["probability"] - row["meta"]["oracle"])) for row in kept]
    strata = defaultdict(list)
    for row in rows:
        strata[reasoning_state(row)].append(row)
    return {"n_planned": len(rows), "n_observed": sum(row["observed"] for row in rows),
            "n_valid": len(kept), "valid_rate": len(kept) / len(rows) if rows else None,
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "analysis_exclusion_counts": dict(Counter(row["analysis_exclusion"] for row in rows if row.get("analysis_exclusion"))),
            "mae": estimate(errors), "signed_residual": estimate([(row["meta"]["family_id"], row["probability"] - row["meta"]["oracle"]) for row in kept]),
            "endpoint_counts": {"exact_zero": sum(row["probability"] == 0 for row in kept),
                                "exact_one": sum(row["probability"] == 1 for row in kept)},
            "output_tokens": token_summary(rows), "reasoning_tokens": token_summary(rows, "reasoning_tokens", True),
            "reasoning_state_counts": dict(Counter(reasoning_state(row) for row in rows)),
            "finish_reason_counts": dict(Counter(str(row.get("recorded_metadata", {}).get("finish_reason", "unknown")) for row in rows if row["observed"])),
            "reasoning_strata": {state: {"n_observed": sum(row["observed"] for row in members),
                                          "n_valid_for_mae": sum(valid(row) for row in members),
                                          "mae": estimate([(row["meta"]["family_id"], abs(row["probability"] - row["meta"]["oracle"])) for row in members if valid(row)])}
                                  for state, members in sorted(strata.items())}}


def linear_bounds(units, coefficients, directional=False, subtract_oracle=False, error=False, prior=False):
    """Sharp bounds for this linear mean, retaining identity of reused records.

    For an MAE contrast the unknown variable is |p-oracle| in
    [0,max(oracle,1-oracle)]. Each metric is bounded separately.
    """
    if not units:
        return {"lower": None, "upper": None, "reason": "no structurally complete planned units"}
    family_counts = Counter(unit["family"] for unit in units)
    terms, records, constant = defaultdict(float), {}, 0.0
    for unit in units:
        weight = 1 / len(family_counts) / family_counts[unit["family"]]
        direction = (2 * unit["answer"] - 1) if directional else 1
        for name, coefficient in coefficients.items():
            row = unit["cells"][name]
            identity = (row["model"], row["item_id"])
            scaled = weight * coefficient * direction
            terms[identity] += scaled
            records[identity] = row
            if subtract_oracle:
                constant -= scaled * row["meta"]["oracle"]
            if prior:
                constant -= scaled * row["meta"]["prior"]
    lower = upper = constant
    missing, nonzero_missing = 0, 0
    for identity, coefficient in terms.items():
        row = records[identity]
        if valid(row):
            value = abs(row["probability"] - row["meta"]["oracle"]) if error else row["probability"]
            lower += coefficient * value
            upper += coefficient * value
        else:
            missing += 1
            nonzero_missing += abs(coefficient) > 1e-15
            limit = max(row["meta"]["oracle"], 1 - row["meta"]["oracle"]) if error else 1
            lower += min(0, coefficient * limit)
            upper += max(0, coefficient * limit)
    return {"lower": lower, "upper": upper, "n_planned_units": len(units),
            "n_families": len(family_counts), "n_unique_records": len(records),
            "n_unknown_unique_records": missing, "n_unknown_with_nonzero_coefficient": nonzero_missing,
            "method": "sharp finite-grid bounds: valid probabilities fixed; each distinct missing/invalid record has one unknown p in [0,1]; not a confidence interval"}


def grouped_sets(rows, fields, classify):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[canonical([row["meta"][key] for key in fields])][classify(row["meta"])].append(row)
    return list(grouped.values())


def contrast(name, groups, coefficients, directional=True, error_contrast=True,
             template_breakdown=True, relation_breakdown=True):
    required = tuple(coefficients)
    counts = dict(candidate_sets=len(groups), complete_valid_sets=0, missing_cell_sets=0,
                  duplicate_cell_sets=0, invalid_or_unobserved_sets=0, incompatible_metadata_sets=0)
    units, kept, missing, invalid = [], [], Counter(), Counter()
    for group in groups:
        absent = [cell for cell in required if cell not in group]
        if absent:
            counts["missing_cell_sets"] += 1
            missing.update(absent)
            continue
        if any(len(group[cell]) != 1 for cell in required):
            counts["duplicate_cell_sets"] += 1
            continue
        cells = {cell: group[cell][0] for cell in required}
        first = cells[required[0]]["meta"]
        if len({r["meta"]["family_id"] for r in cells.values()}) != 1 or (directional and first["answer"] not in (0, 1)):
            counts["incompatible_metadata_sets"] += 1
            continue
        unit = {"family": first["family_id"], "template": first["template_id"],
                "answer": first["answer"], "cells": cells}
        units.append(unit)
        bad = [cell for cell, row in cells.items() if not valid(row)]
        if bad:
            counts["invalid_or_unobserved_sets"] += 1
            invalid.update(bad)
        else:
            kept.append(unit)
            counts["complete_valid_sets"] += 1
    values = defaultdict(list)
    for unit in kept:
        delta = sum(coefficient * unit["cells"][cell]["probability"] for cell, coefficient in coefficients.items())
        normative = sum(coefficient * unit["cells"][cell]["meta"]["oracle"] for cell, coefficient in coefficients.items())
        metrics = {"signed_probability_effect": delta, "absolute_effect": abs(delta),
                   "fraction_absolute_gt_0_02": float(abs(delta) > .02),
                   "fraction_absolute_gt_0_05": float(abs(delta) > .05),
                   "normative_probability_effect": normative, "probability_residual": delta - normative}
        if directional:
            direction = 2 * unit["answer"] - 1
            metrics.update(signed_directional_effect=delta * direction,
                           normative_directional_effect=normative * direction,
                           directional_residual=(delta - normative) * direction)
        if error_contrast:
            metrics["mae_contrast"] = sum(coefficient * abs(unit["cells"][cell]["probability"] - unit["cells"][cell]["meta"]["oracle"])
                                           for cell, coefficient in coefficients.items())
        for key, value in metrics.items():
            values[key].append((unit["family"], value))
    bounds = {"signed_probability_effect": linear_bounds(units, coefficients),
              "probability_residual": linear_bounds(units, coefficients, subtract_oracle=True)}
    if directional:
        bounds["signed_directional_effect"] = linear_bounds(units, coefficients, directional=True)
        bounds["directional_residual"] = linear_bounds(units, coefficients, directional=True, subtract_oracle=True)
    if error_contrast:
        bounds["mae_contrast"] = linear_bounds(units, coefficients, error=True)
    structurally_complete = not any(counts[key] for key in ("missing_cell_sets", "duplicate_cell_sets", "incompatible_metadata_sets"))
    if not structurally_complete:
        bounds = {key: {"lower": None, "upper": None, "reason": "manifest lacks an unambiguous complete structural set; do not claim full-grid bounds"} for key in bounds}
    result = {"name": name, "coefficients": coefficients, "counts": counts,
              "missing_cells": dict(missing), "invalid_or_unobserved_cells": dict(invalid),
              "metrics": {key: estimate(value) for key, value in sorted(values.items())},
              "maximum_absolute_effect": max((abs(sum(coefficients[key] * row["probability"] for key, row in u["cells"].items())) for u in kept), default=None),
              "missing_probability_bounds": bounds,
              "n_unique_records_referenced": len({(r["model"], r["item_id"]) for u in units for r in u["cells"].values()}),
              "normative_full_planned_grid": estimate([(u["family"], sum(coefficients[key] * row["meta"]["oracle"] for key, row in u["cells"].items()) * ((2 * u["answer"] - 1) if directional else 1)) for u in units]),
              "normative_full_planned_grid_is_directional": directional}
    if template_breakdown:
        templates = sorted({r["meta"]["template_id"] for group in groups for members in group.values() for r in members})
        result["by_template"] = {
            template: contrast(name + ":" + template,
                               [{cell: members for cell, members in group.items() if members[0]["meta"]["template_id"] == template}
                                for group in groups if any(members[0]["meta"]["template_id"] == template for members in group.values())],
                               coefficients, directional, error_contrast, False, False)
            for template in templates}
    if relation_breakdown:
        relational = defaultdict(list)
        for group in groups:
            visible = [r for members in group.values() for r in members
                       if r["meta"]["protocol"] in ("zero", "independent")
                       and r["meta"]["visibility"] == "signal_visible"]
            if visible:
                labels = {"agree" if r["meta"]["signal"] == r["meta"]["answer"] else "conflict" for r in visible}
                if len(labels) != 1:
                    raise ValueError("One contrast unit contains incompatible agreement relations")
                relational[next(iter(labels))].append(group)
        if relational:
            result["by_relation"] = {
                relation: contrast(name + ":" + relation, subgroup, coefficients,
                                   directional, error_contrast, template_breakdown, False)
                for relation, subgroup in sorted(relational.items())}
    return result


def hidden_updates(rows):
    groups = defaultdict(list)
    for row in rows:
        if "mechanism" in row["meta"]["analysis_modules"] and row["meta"]["visibility"] == "signal_hidden":
            key = (row["model"], row["meta"]["protocol"], row["meta"]["attribution"], row["meta"]["template_id"])
            groups[key].append(row)
    result = []
    for key, members in sorted(groups.items()):
        metrics = defaultdict(list)
        for row in members:
            if not valid(row):
                continue
            m = row["meta"]
            observed = (row["probability"] - m["prior"]) * (2 * m["answer"] - 1)
            normative = (m["oracle"] - m["prior"]) * (2 * m["answer"] - 1)
            for name, value in (("directional_update_from_prior", observed), ("normative_directional_update", normative), ("update_residual", observed - normative)):
                metrics[name].append((m["family_id"], value))
        units = [{"family": row["meta"]["family_id"], "answer": row["meta"]["answer"], "cells": {"response": row}} for row in members]
        result.append({**dict(zip(("model", "protocol", "attribution", "template_id"), key)), **describe(members),
                       "metrics": {name: estimate(values) for name, values in metrics.items()},
                       "missing_probability_bounds": {
                           "directional_update_from_prior": linear_bounds(units, {"response": 1}, directional=True, prior=True),
                           "update_residual": linear_bounds(units, {"response": 1}, directional=True, subtract_oracle=True)}})
    return result


def mechanism_analyses(rows):
    output = []
    for model in sorted({row["model"] for row in rows}):
        members = [row for row in rows if row["model"] == model and "mechanism" in row["meta"]["analysis_modules"]]
        if not members:
            continue
        attribution = []
        for protocol in PROTOCOLS:
            for visibility in ("signal_hidden", "signal_visible", "answer_hidden"):
                selected = [r for r in members if r["meta"]["protocol"] == protocol and r["meta"]["visibility"] == visibility]
                if not selected:
                    continue
                for left, right in (("self", "other"), ("self", "neutral"), ("other", "neutral")):
                    subset = [r for r in selected if r["meta"]["attribution"] in (left, right)]
                    groups = grouped_sets(subset, WORLD_FIELDS + ("signal", "answer", "reminder", "output_instruction"), lambda m: m["attribution"])
                    attribution.append(contrast(f"{protocol}:{visibility}:{left}_minus_{right}", groups,
                                                {left: 1, right: -1}, visibility != "answer_hidden"))
        increments = []
        for protocol in PROTOCOLS:
            for attributed in ATTRIBUTIONS:
                selected = [r for r in members if r["meta"]["protocol"] == protocol and r["meta"]["attribution"] == attributed]
                hidden = defaultdict(list)
                for row in selected:
                    if row["meta"]["visibility"] == "answer_hidden":
                        hidden[tuple(row["meta"][k] for k in WORLD_FIELDS + ("signal", "reminder", "output_instruction"))].append(row)
                groups = []
                for row in selected:
                    if row["meta"]["visibility"] != "signal_visible":
                        continue
                    key = tuple(row["meta"][k] for k in WORLD_FIELDS + ("signal", "reminder", "output_instruction"))
                    group = {"visible": [row]}
                    if hidden[key]:
                        group["answer_hidden"] = hidden[key]
                    groups.append(group)
                if groups:
                    result = contrast(f"{protocol}:{attributed}:answer_visible_minus_hidden", groups,
                                      {"visible": 1, "answer_hidden": -1}, True, False)
                    result["matching_rule"] = "Each legal S-visible (S,A) row references the same-protocol, same-S, same-attribution, world/context/template A-hidden row. Zero/independent reuse one anchor for two A values; references do not create raw observations."
                    increments.append(result)
        output.append({"model": model, "n_unique_mechanism_items": len(members),
                       "attribution": attribution, "answer_visibility_increments": increments})
    return output


def wording_analyses(rows):
    results = []
    for model in sorted({r["model"] for r in rows}):
        selected = [r for r in rows if r["model"] == model and "wording" in r["meta"]["analysis_modules"]]
        if not selected:
            continue
        analyses = []
        for protocol in ("positive", "independent"):
            members = [r for r in selected if r["meta"]["protocol"] == protocol]
            key_fields = WORLD_FIELDS + ("signal", "answer")
            factors = ("reminder", "output_instruction", "attribution")
            levels = (REMINDERS, INSTRUCTIONS, ("self", "other"))
            for varied in range(3):
                fixed = [i for i in range(3) if i != varied]
                for one in levels[fixed[0]]:
                    for two in levels[fixed[1]]:
                        subset = [r for r in members if r["meta"][factors[fixed[0]]] == one and r["meta"][factors[fixed[1]]] == two]
                        groups = grouped_sets(subset, key_fields, lambda m, f=factors[varied]: m[f])
                        analyses.append(contrast(f"{protocol}:{factors[varied]}:{one}:{two}", groups,
                                                 {levels[varied][0]: 1, levels[varied][1]: -1}))
            groups = grouped_sets(members, key_fields, lambda m: ":".join(m[k] for k in factors))
            for name, indexes, divisor in (
                ("reminder_main", (0,), 4), ("instruction_main", (1,), 4), ("attribution_main", (2,), 4),
                ("reminder_by_attribution", (0, 2), 2), ("instruction_by_attribution", (1, 2), 2),
                ("reminder_by_instruction", (0, 1), 2), ("three_way_interaction", (0, 1, 2), 1)):
                coefficients = {}
                for reminder in REMINDERS:
                    for instruction in INSTRUCTIONS:
                        for attributed in ("self", "other"):
                            cell = (reminder, instruction, attributed)
                            sign = math.prod(1 if cell[i] == levels[i][0] else -1 for i in indexes)
                            coefficients[":".join(cell)] = sign / divisor
                analyses.append(contrast(f"{protocol}:{name}", groups, coefficients))
        results.append({"model": model, "n_unique_wording_items_referenced": len(selected),
                        "n_shared_with_mechanism": sum("mechanism" in r["meta"]["analysis_modules"] for r in selected),
                        "factor_signs": {"reminder": "scoped minus legacy", "instruction": "compatible minus restrictive", "attribution": "self minus other"},
                        "factorial_policy": "Main effects average the four levels of other factors; two-way interactions average the two levels of the third factor; three-way is the unscaled difference of differences of differences. Complete 8-cell sets only for factorial point estimates.",
                        "analyses": analyses})
    return results


def gate_qualification(rows, quality_details):
    output = []
    for model in sorted({r["model"] for r in rows}):
        members = [r for r in rows if r["model"] == model and r["meta"]["module"] == "gate"]
        if not members:
            continue
        groups = []
        for protocol in ("positive", "zero", "negative", "baseline"):
            stats = describe([r for r in members if r["meta"]["protocol"] == protocol])
            mae = stats["mae"]["estimate"]
            passes = stats["n_planned"] == 16 and stats["n_observed"] == 16 and stats["valid_rate"] >= VALID_GATE and mae is not None and mae <= MAE_GATE
            groups.append({"protocol": protocol, **stats, "mae_family_equal": mae, "passes": passes})
        complete = all(g["n_observed"] == g["n_planned"] == 16 for g in groups)
        quality_ok = not any(d["model"] == model for d in quality_details)
        qualifies = complete and quality_ok and all(g["passes"] for g in groups)
        output.append({"model": model, "decision": "qualified" if qualifies else ("incomplete" if not complete else "not_qualified"),
                       "all_criteria_passed": qualifies, "data_quality_clean": quality_ok,
                       "criteria": {"n_required_per_group": 16, "n_required_total": 64,
                                    "minimum_valid_rate_each_group": VALID_GATE, "maximum_family_mae_each_group": MAE_GATE,
                                    "all_records_terminal_required": True,
                                    "interpretation": "At 16 items per group the 95% threshold requires 16/16 valid; conservative capability qualification, never selection by attribution effect."},
                       "groups": groups})
    return output


def render_report(summary):
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    lines = ["# I16 机制区分：固定设计分析", "", "生成时间：" + summary["generated_at"], "",
             "实际记录只计一次，共享的mechanism/wording行仅在分析中复用。数值为family等权有限网格描述；本轮仅两数值family，不计算bootstrap、置信区间或p值。",
             "无效、截断及未观测均保留计数。数值点估计针对有效完整配对；缺失界固定所有有效值、让每条不同缺失记录的概率取[0,1]，不是置信区间。",
             "空思考是生成后的诊断，未用于筛选主样本，不能作为已识别的心理机制或因果中介。", "",
             "## 完成情况", "", "| 模型 | 已观测/计划 | 有效 | 状态 |", "|---|---:|---:|---|"]
    for model in summary["models"]:
        lines.append(f"| {model['model']} | {model['n_observed']}/{model['n_planned']} | {model['n_valid']} | {canonical(model['status_counts'])} |")
    lines += ["", "元数据/归属检查：`" + canonical(summary["data_quality_issues"]) + "`。", "",
              "## 无归属能力资格检查", "", "| 模型 | 结果 | 组 | 有效/计划 | MAE | 组通过 |", "|---|---|---|---:|---:|---|"]
    for gate in summary["gate_qualification"]:
        for group in gate["groups"]:
            lines.append(f"| {gate['model']} | {gate['decision']} | {group['protocol']} | {group['n_valid']}/{group['n_planned']} | {fmt(group['mae_family_equal'])} | {group['passes']} |")
    lines += ["", "四类各16题分别要求有效率≥95%和MAE≤0.05，因此各组实际必须16条均有效；仅此能力门槛用于决定预定配置是否进入main。", "",
              "## S隐藏：相对先验的方向更新", "", "| 模型/协议/归属/模板 | 有效/计划 | 实际方向更新 | 规范更新 | 残差 |", "|---|---:|---:|---:|---:|"]
    for group in summary["hidden_updates"]:
        get = lambda key: fmt(group["metrics"].get(key, {}).get("estimate"))
        lines.append(f"| {group['model']}/{group['protocol']}/{group['attribution']}/{group['template_id']} | {group['n_valid']}/{group['n_planned']} | {get('directional_update_from_prior')} | {get('normative_directional_update')} | {get('update_residual')} |")
    lines += ["", "## 主要配对对照", "", "方向取A的方向；A隐藏时仅报告有符号概率差，不虚构A方向。MAE contrast在self−other对照中就是MAE_self−MAE_other。", "",
              "| 模型/对照 | 完整/候选 | 方向差（或A隐藏概率差） | 平均绝对差 | MAE contrast | 缺失最坏界 |", "|---|---:|---:|---:|---:|---|"]
    collections = [(m["model"], c) for m in summary["mechanism"] for c in m["attribution"] + m["answer_visibility_increments"]]
    collections += [(m["model"], c) for m in summary["wording"] for c in m["analyses"]]
    for model, comparison in collections:
        metric = "signed_directional_effect" if comparison["normative_full_planned_grid_is_directional"] else "signed_probability_effect"
        metrics, counts = comparison["metrics"], comparison["counts"]
        bounds = comparison["missing_probability_bounds"][metric]
        lines.append(f"| {model}/{comparison['name']} | {counts['complete_valid_sets']}/{counts['candidate_sets']} | {fmt(metrics.get(metric, {}).get('estimate'))} | {fmt(metrics.get('absolute_effect', {}).get('estimate'))} | {fmt(metrics.get('mae_contrast', {}).get('estimate'))} | [{fmt(bounds.get('lower'))}, {fmt(bounds.get('upper'))}] |")
    lines += ["", "各完整条件MAE、状态、逐模板效应、最大绝对差、超阈值比例、共享锚点和空思考分布完整保存在summary.json。",
              "只对每个协议的合法观测做匹配。A隐藏锚点可被多个A值引用，但保持同一原始记录及同一缺失未知值；未把S隐藏与S显示的不同合法组合强行制造visibility交互。",
              "本轮仅概率JSON行为读出；真实历史自生成、心理自我承诺、logit/action/surprisal及外部泛化均未由此分析验证。", ""]
    return "\n".join(lines)


def analyze(db, output_dir, manifest_path, models=None):
    manifest, manifest_hash = load_manifest(manifest_path)
    rows, issues, quality_details, stored = read_snapshot(db, manifest, models)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"],) + tuple(row["meta"][key] for key in GROUP_FIELDS)].append(row)
    model_summaries = []
    for model in sorted({r["model"] for r in rows}):
        members = [r for r in rows if r["model"] == model]
        model_summaries.append({"model": model, **describe(members),
                                "n_database_rows": sum(str(r["model"]) == model for r in stored),
                                "n_unique_valid_probabilities": len({r["probability"] for r in members if valid(r)})})
    summary = {"schema_version": 1, "analysis_kind": "i16-mechanism-finite-grid-v1",
               "generated_at": datetime.now(timezone.utc).isoformat(), "database": str(Path(db).resolve()),
               "manifest": str(Path(manifest_path).resolve()), "manifest_sha256": manifest_hash,
               "analysis_source_sha256": digest(__file__), "n_database_rows": len(stored),
               "n_manifest_items": len(manifest), "physical_module_counts": dict(Counter(m["module"] for m in manifest.values())),
               "logical_module_reference_counts": {module: sum(module in m["analysis_modules"] for m in manifest.values()) for module in ("gate", "mechanism", "wording")},
               "n_design_families": len({m["family_id"] for m in manifest.values()}),
               "estimand": "Equal finite numerical-family means; correlated values, templates and reused references are not IID samples",
               "uncertainty": "No bootstrap, CI or p-values; missing-probability intervals are deterministic sharp bounds for the specified linear estimand",
               "metadata_policy": "All manifest fields must exactly match stored metadata. Manifest provides intended cell coordinates only; mismatching recorded metadata are retained as exclusions with no repaired probability. Unknown and duplicate items cannot become valid analysis rows.",
               "data_quality_issues": issues, "data_quality_details": quality_details,
               "models": model_summaries,
               "groups": [{**dict(zip(("model",) + GROUP_FIELDS, key)), **describe(members)} for key, members in sorted(groups.items(), key=lambda x: canonical(x[0]))],
               "gate_qualification": gate_qualification(rows, quality_details),
               "hidden_updates": hidden_updates(rows), "mechanism": mechanism_analyses(rows),
               "wording": wording_analyses(rows)}
    output = Path(output_dir)
    atomic_write(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(output / "report.md", render_report(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", help="Expected model names, including wholly unobserved configurations")
    args = parser.parse_args()
    summary = analyze(args.db, args.output_dir, args.manifest, args.models)
    print(json.dumps({"n_database_rows": summary["n_database_rows"],
                      "data_quality_issues": summary["data_quality_issues"],
                      "gate_qualification": [{"model": g["model"], "decision": g["decision"]} for g in summary["gate_qualification"]],
                      "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))
    return 2 if summary["data_quality_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
