"""Exploratory behavior references from saved outputs. No inference or input edits."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT.parent / "i16-dependence-enhancement"
TOLERANCES = (0.0001, 0.0005)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def posterior(prior, reliability, observations, rho=None):
    pi, r = Fraction(str(prior)), Fraction(str(reliability))
    likelihoods = []
    for h in (0, 1):
        first = r if observations[0] == h else 1 - r
        if len(observations) == 1:
            likelihoods.append(first)
            continue
        second = r if observations[1] == h else 1 - r
        if rho is not None:
            t = Fraction(str(rho))
            second = t * int(observations[0] == observations[1]) + (1 - t) * second
        likelihoods.append(first * second)
    denominator = pi * likelihoods[1] + (1 - pi) * likelihoods[0]
    assert denominator > 0
    return float(pi * likelihoods[1] / denominator)


def summarize(rows):
    valid = [x for x in rows if x["status"] == "ok"]
    family_errors = {}
    for name in ("joint", "first_only", "last_only", "independent_pair"):
        families = defaultdict(list)
        for x in valid:
            families[x["family_id"]].append(abs(x["probability"] - x[name]))
        family_errors[name] = {
            "mean_on_valid": mean(abs(x["probability"] - x[name]) for x in valid) if valid else None,
            "n_families": len(families),
            "equal_family_mean": mean(mean(v) for v in families.values()) if len(families) == 6 else None,
        }
    return {
        "planned": len(rows),
        "statuses": dict(Counter(x["status"] for x in rows)),
        "matching_sets": {
            str(t): dict(Counter(
                "|".join(sorted(name for name in ("joint", "first_only", "last_only", "independent_pair")
                                if abs(x["probability"] - x[name]) <= t)) or "unmatched"
                if x["status"] == "ok" else "missing"
                for x in rows)) for t in TOLERANCES
        },
        "reference_errors": family_errors,
    }


def main():
    source_paths = [INPUT / "DESIGN_SEAL.json", INPUT / "manifests/partial.jsonl",
                    INPUT / "inputs/original_controls.jsonl", INPUT / "queues/phi/merged.sqlite",
                    INPUT / "queues/qwen/merged.sqlite"]
    before = {p.relative_to(INPUT).as_posix(): sha(p) for p in source_paths}
    items = read_lines(INPUT / "manifests/partial.jsonl")
    assert len(items) == 1344 and len({x["id"] for x in items}) == 1344
    records = {}
    for name in ("phi", "qwen"):
        with sqlite3.connect((INPUT / f"queues/{name}/merged.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            for record in db.execute("SELECT * FROM results"):
                record = dict(record)
                assert record["item_id"] not in records
                records[record["item_id"]] = record
    rows, groups, directional = [], defaultdict(list), defaultdict(list)
    for item in items:
        record = records[item["id"]]
        metadata = json.loads(record["metadata_json"])
        assert all(metadata.get(k) == v for k, v in item.items())
        b = item["next_signal"]
        q = posterior(item["prior"], item["reliability"], [item["signal"]] if b is None
                      else [item["signal"], b], None if b is None else item["rho"])
        assert abs(q - item["oracle"]) < 1e-12
        if b is None:
            continue
        row = {k: item[k] for k in ("id", "family_id", "parent_item_id", "history_arm", "rho", "branch",
                                    "prior", "reliability", "signal", "next_signal")}
        row.update(model=item["model_scope"][0], status=record["status"],
                   probability=record["probability"] if record["status"] == "ok" else None,
                   joint=q,
                   first_only=posterior(item["prior"], item["reliability"], [item["signal"]]),
                   last_only=posterior(item["prior"], item["reliability"], [b]),
                   independent_pair=posterior(item["prior"], item["reliability"], [item["signal"], b]))
        rows.append(row)
        key = (row["model"], row["history_arm"], row["rho"], row["branch"])
        groups[key].append(row)
        if row["branch"] == "conflict":
            directional[key + (row["signal"],)].append(row)
    old = read_lines(INPUT / "inputs/original_controls.jsonl")
    endpoint_results = []
    for rho, kind in ((0, "independent"), (1, "copy")):
        oldmap = {x["source_item"]["parent_item_id"]: x for x in old
                  if x["source_item"]["history_arm"] == "native_self"
                  and x["source_item"]["evidence_kind"] == kind
                  and x["source_item"]["next_signal"] == x["source_item"]["signal"]}
        paired = []
        for x in rows:
            if x["model"] != "phi4-standard-r6" or x["history_arm"] != "native_self" or x["rho"] != rho or x["branch"] != "agree":
                continue
            original = oldmap[x["parent_item_id"]]
            new_item = next(t for t in items if t["id"] == x["id"])
            assert all(new_item[k] == original["source_item"][k] for k in (
                "prior", "reliability", "signal", "family_id", "parent_final_sha256", "parent_record_sha256"))
            op = original["source_record"]["probability"]
            assert original["source_record"]["status"] == x["status"] == "ok"
            paired.append({"old_id": original["source_item"]["id"], "new_id": x["id"],
                           "old": op, "new": x["probability"], "first_only": x["first_only"],
                           "independent_pair": x["independent_pair"]})
        assert len(paired) == 24
        endpoint_results.append({"rho": rho, "n_matched_parents": len(paired), "pairs": paired,
                                 "reference_mae": {f"{version}_{reference}": mean(abs(x[version] - x[reference]) for x in paired)
                                                   for version in ("old", "new") for reference in ("first_only", "independent_pair")}})
    after = {p.relative_to(INPUT).as_posix(): sha(p) for p in source_paths}
    assert before == after
    result = {
        "analysis_type": "exploratory post hoc numerical reference comparison",
        "inference_started": False, "original_inputs_unchanged": True, "input_sha256": before,
        "tolerances": list(TOLERANCES),
        "interpretation": "Reference matches describe reported probabilities, not internal algorithms. Matching references can coincide. Repeated rho conditions reuse parents and are not independent samples.",
        "groups": [{"model": k[0], "history_arm": k[1], "rho": k[2], "branch": k[3], **summarize(v)}
                   for k, v in sorted(groups.items())],
        "conflict_by_first_reading": [{"model": k[0], "history_arm": k[1], "rho": k[2], "branch": k[3], "signal": k[4], **summarize(v)}
                                     for k, v in sorted(directional.items())],
        "matched_endpoints": endpoint_results,
    }
    destination = ROOT / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "BEHAVIOR_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "reference_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "groups": len(groups), "matched_endpoints": [
        {k: v for k, v in x.items() if k != "pairs"} for x in endpoint_results],
        "original_inputs_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
