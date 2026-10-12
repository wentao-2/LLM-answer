"""Meaningful synthetic checks for finite-design framing analysis; no model calls."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import analyze_formal as analysis
from build_formal import build_formal_rows


def write_database(path, manifest, response, omitted=(), truncated=()):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE results (id TEXT PRIMARY KEY, model TEXT, item_id TEXT,
                   status TEXT, probability REAL, raw_output TEXT, input_tokens INTEGER,
                   output_tokens INTEGER, duration_sec REAL, created_at TEXT, metadata_json TEXT)""")
    for item in manifest:
        if item["id"] in omitted:
            continue
        value = response(item)
        status = "truncated" if item["id"] in truncated else "ok"
        con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("test:" + item["id"], "test", item["id"], status, value, json.dumps({"p_h1": value}),
                     1, 1, .01, "synthetic", json.dumps(item)))
    con.commit()
    con.close()


def find_analysis(summary, arm, suffix):
    arms = summary["core"][0]["arms"]
    return next(a for collection in arms if collection["reminder"] == arm
                for a in collection["analyses"] if a["name"].endswith(":" + suffix))


class FormalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = build_formal_rows()
        cls.small = [r for r in cls.full if r["prior"] in (.45, .55) and r["reliability"] in (.60, .70)]

    def run_analysis(self, rows, response=lambda row: row["oracle"], omitted=(), truncated=(), reps=0):
        temporary = tempfile.TemporaryDirectory(prefix="i16-formal-analysis-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        db = root / "results.sqlite"
        write_database(db, rows, response, omitted, truncated)
        return analysis.analyze(db, root / "analysis", manifest, reps, 42), root

    def test_full_oracle_no_cloned_counts_or_duplicate_prompts(self):
        summary, _ = self.run_analysis(self.full)
        self.assertEqual(summary["n_database_rows"], 3480)
        self.assertEqual(summary["models"][0]["n_valid"], 3480)
        self.assertEqual(summary["models"][0]["n_design_families"], 15)
        self.assertEqual(sum(g["n_observed"] for g in summary["groups"]), 3480)
        self.assertEqual(summary["exact_prompt_duplicates"]["n_duplicate_prompt_groups"], 0)
        self.assertFalse(summary["data_quality_issues"])
        for arm in summary["core"][0]["arms"]:
            self.assertEqual(arm["shared_baseline_records_referenced"], 240)
            for comparison in arm["analyses"]:
                self.assertEqual(comparison["counts"]["complete_valid_sets"], comparison["counts"]["candidate_sets"])
                self.assertEqual(comparison["n_families"], 15)
        for g in summary["groups"]:
            self.assertEqual(g["mae"]["estimate"], 0)
        for arm in ("explicit", "minimal"):
            contrast = find_analysis(summary, arm, "baseline_update_independent_conflict_self")
            self.assertEqual(contrast["counts"]["complete_valid_sets"], 240)
            self.assertEqual(contrast["metrics"]["update_residual"]["estimate"], 0)
            self.assertGreater(contrast["metrics"]["normative_directional_update"]["estimate"], 0)

    def test_known_label_symmetric_effects_and_reminder_interaction(self):
        def response(row):
            if row["attribution"] != "self":
                return row["oracle"]
            magnitude = .04 if row["reminder"] == "explicit" else .01
            magnitude *= 2 if row["condition"] == "independent" else 1
            return row["oracle"] + magnitude * (2 * row["answer"] - 1)

        summary, _ = self.run_analysis(self.small, response, reps=20)
        same = find_analysis(summary, "explicit", "attribution_same_all")
        self.assertAlmostEqual(same["metrics"]["signed_directional_effect"]["estimate"], .04)
        self.assertAlmostEqual(same["metrics"]["absolute_effect"]["estimate"], .04)
        self.assertEqual(same["metrics"]["fraction_absolute_gt_0_02"]["estimate"], 1)
        self.assertEqual(same["metrics"]["fraction_absolute_gt_0_05"]["estimate"], 0)
        self.assertEqual(same["n_families"], 2)
        self.assertEqual(same["n_worlds"], 8)
        self.assertAlmostEqual(same["maximum_absolute_effect"], .04)
        independent = find_analysis(summary, "explicit", "attribution_independent_conflict")
        self.assertAlmostEqual(independent["metrics"]["signed_directional_effect"]["estimate"], .08)
        self.assertEqual(independent["metrics"]["fraction_absolute_gt_0_05"]["estimate"], 1)
        interaction = find_analysis(summary, "explicit", "attribution_by_dependence_interaction")
        self.assertAlmostEqual(interaction["metrics"]["self_minus_other_dependence_sensitivity"]["estimate"], .04)
        comparisons = summary["core"][0]["paired_reminder_comparisons"]
        reminder = next(c for c in comparisons if c["name"].endswith(":reminder_effect_on_attribution_same_all"))
        self.assertAlmostEqual(reminder["metrics"]["signed_directional_effect"]["estimate"], .03)
        for template in same["by_template"].values():
            self.assertAlmostEqual(template["metrics"]["signed_directional_effect"]["estimate"], .04)

    def test_missing_truncated_and_constant_response_are_visible(self):
        targets = [r for r in self.small if r["reminder"] == "explicit" and r["condition"] == "same"
                   and r["attribution"] == "self"]
        omitted, truncated = targets[0]["id"], targets[1]["id"]
        summary, _ = self.run_analysis(self.small, lambda row: .123, (omitted,), (truncated,))
        model = summary["models"][0]
        self.assertEqual(model["status_counts"]["unobserved"], 1)
        self.assertEqual(model["status_counts"]["truncated"], 1)
        self.assertEqual(model["n_unique_valid_probabilities"], 1)
        self.assertEqual(model["dominant_probability"], .123)
        same = find_analysis(summary, "explicit", "attribution_same_all")
        self.assertEqual(same["counts"]["invalid_or_unobserved_sets"], 2)
        self.assertEqual(same["counts"]["complete_valid_sets"], same["counts"]["candidate_sets"] - 2)

    def test_family_weighting_and_deterministic_bootstrap(self):
        observations = [("family_a", 0), ("family_a", 0), ("family_b", 1)]
        first = analysis.estimate(observations, "weight", 100, 42)
        second = analysis.estimate(observations, "weight", 100, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], .5)
        self.assertEqual(first["n_families"], 2)
        self.assertNotIn("n_worlds", first)

    def test_exact_duplicate_items_separate_from_shared_references(self):
        manifest = {"one": {"prompt": "same"}, "two": {"prompt": "same"}, "three": {"prompt": "different"}}
        result = analysis.duplicate_prompts([], manifest)
        self.assertEqual(result["n_duplicate_prompt_groups"], 1)
        self.assertEqual(result["n_items_in_duplicate_groups"], 2)

    def test_reasoning_segment_empty_whitespace_optional_open_and_unclosed(self):
        for raw in ("<think></think>", "<think> \n\t </think>\n{}", " \n </think>\n{}"):
            segment = analysis.reasoning_segment(raw)
            self.assertTrue(segment["closed"])
            self.assertFalse(segment["nonempty"])
            self.assertEqual(segment["state"], "empty")
        for raw in ("<think>Use Bayes.\n</think>\n{}", "Use Bayes.\n</think>\n{}"):
            segment = analysis.reasoning_segment(raw)
            self.assertTrue(segment["nonempty"])
            self.assertEqual(segment["state"], "nonempty")
        self.assertEqual(analysis.reasoning_segment("<think>unfinished")["state"], "unclosed")
        self.assertFalse(analysis.thinking_enabled({"meta": {"model_mode": "nonthinking-explicit-false-direct-json"},
                                                  "raw_output": "<think></think>"}))

    def test_readout_diagnostics_do_not_change_primary_samples(self):
        original, root = self.run_analysis(self.small)
        connection = sqlite3.connect(root / "results.sqlite")
        stored = connection.execute("SELECT id, probability, metadata_json FROM results ORDER BY id").fetchall()
        for index, (item_id, value, encoded) in enumerate(stored):
            meta = json.loads(encoded)
            meta.update(model_mode="thinking-explicit-true-final-json", finish_reason="eos")
            nonempty = index % 2 == 1
            body = "Apply the stated rule." if nonempty else " \n\t "
            raw = "<think>" + body + "</think>\n" + json.dumps({"p_h1": value})
            # Test legacy rows with no extra metadata, and future token metadata.
            if index % 3 == 0:
                meta["reasoning_nonempty"] = nonempty
                meta["reasoning_tokens"] = 5 if nonempty else 0
            connection.execute("UPDATE results SET raw_output=?, output_tokens=?, metadata_json=? WHERE id=?",
                               (raw, 20 if nonempty else 10, json.dumps(meta), item_id))
        connection.commit()
        connection.close()
        updated = analysis.analyze(root / "results.sqlite", root / "new-analysis", root / "manifest.jsonl", 0, 42)
        self.assertEqual(original["n_database_rows"], updated["n_database_rows"])
        self.assertEqual(original["models"], updated["models"])
        self.assertEqual(original["groups"], updated["groups"])
        self.assertEqual(original["core"], updated["core"])
        diagnostics = updated["readout_diagnostics"]
        self.assertEqual(diagnostics["coverage"]["enabled_records"], len(self.small))
        self.assertEqual(sum(g["n_closed"] for g in diagnostics["groups"]), len(self.small))
        self.assertEqual(sum(g["n_empty"] for g in diagnostics["groups"]), len(self.small) // 2)
        self.assertEqual(sum(g["n_nonempty"] for g in diagnostics["groups"]), len(self.small) // 2)
        self.assertEqual(sum(g["reasoning_nonempty_metadata_mismatches"] for g in diagnostics["groups"]), 0)
        for group in diagnostics["groups"]:
            for state in ("empty", "nonempty"):
                values = group["by_reasoning_state"][state]
                if values["n_valid_for_mae"]:
                    self.assertEqual(values["mae"]["estimate"], 0)
                    self.assertEqual(values["output_tokens"]["mean"], 10 if state == "empty" else 20)


if __name__ == "__main__":
    unittest.main()
