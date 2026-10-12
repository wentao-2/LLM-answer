"""Verify candidate contrasts without creating or freezing a manifest."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import unittest

import build_formal as formal
import build_manifest as pilot


V1_OUTPUT_INSTRUCTION = (
    'Return exactly one JSON object of the form {"p_h1":0.123}, where p_h1 is '
    "your probability that H=1 after the evidence stated above. Use a decimal "
    "number between 0 and 1, not a percentage. Do not include reasoning, "
    "additional keys, or any other text."
)


class FormalCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = formal.build_formal_rows()
        cls.pilot_rows = pilot.build_rows()
        cls.base_rows = pilot.build_rows(split="formal", seed=0, priors=formal.PRIORS,
                                        reliabilities=formal.RELIABILITIES)

    def test_counts_and_unique_actual_observations(self):
        self.assertEqual(len(self.rows), 3480)
        self.assertEqual(len({row["prompt"] for row in self.rows}), 3480)
        self.assertEqual(len({row["id"] for row in self.rows}), 3480)
        self.assertEqual(Counter(row["reminder"] for row in self.rows),
                         {"shared": 600, "explicit": 1440, "minimal": 1440})
        self.assertEqual(Counter(row["condition"] for row in self.rows),
                         {"positive": 120, "zero": 120, "negative": 120,
                          "baseline": 240, "same": 960, "independent": 1920})
        self.assertEqual(len({row["world_id"] for row in self.rows}), 60)
        self.assertEqual(len({row["family_id"] for row in self.rows}), 15)
        self.assertEqual({row["split"] for row in self.rows}, {"formal"})

    def test_only_designated_sentence_is_changed(self):
        indexed = {(row["world_id"], row["template_id"], row["condition"],
                    row["signal"], row["answer"], row["attribution"]): row
                   for row in self.base_rows}
        for row in self.rows:
            source = indexed[(row["world_id"], row["template_id"], row["condition"],
                              row["signal"], row["answer"], row["attribution"])]
            for field in source:
                if field not in ("id", "generator_version", "prompt"):
                    self.assertEqual(row[field], source[field])
            if row["reminder"] in ("explicit", "shared"):
                self.assertEqual(row["prompt"], source["prompt"])
            else:
                full, minimal = formal.REMINDER_SENTENCES[row["template_id"]]
                self.assertEqual(row["prompt"], source["prompt"].replace(full, minimal, 1))
                self.assertEqual(row["prompt"].count(minimal), 1)
                self.assertNotIn(full, row["prompt"])

    def test_legal_worlds_and_self_other_pairing(self):
        pairs = defaultdict(dict)
        for row in self.rows:
            self.assertAlmostEqual(row["oracle"], pilot.oracle_posterior(
                row["stage"], row["condition"], row["prior"], row["reliability"],
                row["signal"], row["answer"]))
            if row["condition"] in ("same", "independent"):
                key = tuple(row[field] for field in ("world_id", "template_id", "condition",
                                                    "signal", "answer", "reminder"))
                pairs[key][row["attribution"]] = row
            else:
                self.assertEqual(row["attribution"], "none")
                self.assertEqual(row["reminder"], "shared")
        self.assertEqual(len(pairs), 1440)
        for pair in pairs.values():
            self.assertEqual(set(pair), {"self", "other"})
            self.assertEqual(pair["self"]["oracle"], pair["other"]["oracle"])
            self.assertEqual(pair["self"]["prompt"],
                             pair["other"]["prompt"].replace("another assistant", "you"))

    def test_reminder_arms_share_pair_and_cluster_identifiers(self):
        arms = defaultdict(dict)
        for row in self.rows:
            if row["reminder"] != "shared":
                key = tuple(row[field] for field in ("world_id", "template_id", "condition",
                                                    "signal", "answer", "attribution"))
                arms[key][row["reminder"]] = row
        self.assertEqual(len(arms), 1440)
        for pair in arms.values():
            self.assertEqual(set(pair), {"explicit", "minimal"})
            self.assertEqual(pair["explicit"]["family_id"], pair["minimal"]["family_id"])

    def test_shared_controls_not_duplicated(self):
        shared = [row for row in self.rows if row["reminder"] == "shared"]
        expected = [row for row in self.base_rows if row["condition"] not in ("same", "independent")]
        self.assertEqual({row["prompt"] for row in shared}, {row["prompt"] for row in expected})
        self.assertEqual(len(shared), len(expected))

    def test_pilot_v1_and_v2_prompts_and_numeric_pairs_are_disjoint(self):
        formal_prompts = {row["prompt"] for row in self.rows}
        pilot_v2_prompts = {row["prompt"] for row in self.pilot_rows}
        # v1 differs from v2 only in its preserved output instruction.
        pilot_v1_prompts = {
            row["prompt"].replace(pilot.OUTPUT_INSTRUCTION, V1_OUTPUT_INSTRUCTION)
            for row in self.pilot_rows
        }
        self.assertFalse(formal_prompts & pilot_v2_prompts)
        self.assertFalse(formal_prompts & pilot_v1_prompts)
        formal_pairs = {(row["prior"], row["reliability"]) for row in self.rows}
        pilot_pairs = {(row["prior"], row["reliability"]) for row in self.pilot_rows}
        self.assertEqual(len(formal_pairs), 30)
        self.assertFalse(formal_pairs & pilot_pairs)

    def test_available_frozen_development_manifests_do_not_overlap(self):
        root = Path(__file__).resolve().parent
        candidates = (root / "results/pilot/manifest.jsonl", root / "manifests/pilot.jsonl",
                      root / "manifests/pilot-format-fix.jsonl")
        found = [path for path in candidates if path.exists()]
        if not found:
            self.skipTest("No frozen development manifest available locally")
        formal_prompts = {row["prompt"] for row in self.rows}
        formal_pairs = {(row["prior"], row["reliability"]) for row in self.rows}
        for path in found:
            with path.open(encoding="utf-8") as stream:
                frozen = [json.loads(line) for line in stream if line.strip()]
            self.assertFalse(formal_prompts & {row["prompt"] for row in frozen})
            self.assertFalse(formal_pairs & {(row["prior"], row["reliability"]) for row in frozen})

    def test_deterministic_order_and_order_only_seed(self):
        same = formal.build_formal_rows()
        other = formal.build_formal_rows(seed=1603)
        self.assertEqual(self.rows, same)
        self.assertNotEqual([row["id"] for row in self.rows], [row["id"] for row in other])
        self.assertEqual({row["id"] for row in self.rows}, {row["id"] for row in other})

    def test_readout_has_no_example_and_ids_include_current_record(self):
        for row in self.rows:
            self.assertTrue(row["prompt"].endswith(pilot.OUTPUT_INSTRUCTION))
            self.assertEqual(row["generator_version"], formal.GENERATOR_VERSION)
            self.assertEqual(row["id"], pilot.digest({k: v for k, v in row.items() if k != "id"}))

    def test_invalid_interventions_are_rejected(self):
        core = next(row for row in self.base_rows if row["condition"] == "same")
        baseline = next(row for row in self.base_rows if row["condition"] == "baseline")
        with self.assertRaises(ValueError):
            formal.with_reminder(core, "shared")
        with self.assertRaises(ValueError):
            formal.with_reminder(baseline, "minimal")
        changed_source = dict(core, prompt=core["prompt"].replace("This attribution", "This label").replace("The author attribution", "The author label"))
        with self.assertRaises(ValueError):
            formal.with_reminder(changed_source, "minimal")


if __name__ == "__main__":
    unittest.main()
