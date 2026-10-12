"""Scientific mechanism, balancing, determinism, and publication checks."""

import collections
import itertools
from pathlib import Path
import tempfile
import unittest

import build_manifest as manifest


class MechanismTests(unittest.TestCase):
    def test_calibration_known_posteriors(self):
        for condition, expected in (("positive", .8), ("zero", .5), ("negative", .2)):
            self.assertAlmostEqual(manifest.oracle_posterior("calibration", condition, .5, .8, None, 1), expected)

    def test_new_information_known_posteriors(self):
        self.assertAlmostEqual(manifest.oracle_posterior("core", "same", .5, .8, 1, 1), .8)
        self.assertAlmostEqual(manifest.oracle_posterior("core", "independent", .5, .8, 1, 1), 16 / 17)
        self.assertAlmostEqual(manifest.oracle_posterior("core", "baseline", .5, .8, 1, None), .8)
        self.assertAlmostEqual(manifest.oracle_posterior("core", "independent", .25, .8, 1, 0), .25)

    def test_label_complement_symmetry(self):
        for prior, reliability in itertools.product(manifest.PILOT_PRIORS, manifest.PILOT_RELIABILITIES):
            for signals in ((), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)):
                original = manifest.probability_h1(prior, reliability, signals)
                flipped = manifest.probability_h1(1 - prior, reliability, tuple(1 - s for s in signals))
                self.assertAlmostEqual(original + flipped, 1)

    def test_illegal_worlds_rejected(self):
        for args in (("core", "same", .5, .8, 1, 0),
                     ("core", "baseline", .5, .8, 1, 1),
                     ("calibration", "positive", .5, .8, 1, 1)):
            with self.assertRaises(ValueError):
                manifest.oracle_posterior(*args)

    def test_mechanism_probability_normalization(self):
        for prior, reliability in itertools.product(manifest.PILOT_PRIORS, manifest.PILOT_RELIABILITIES):
            for condition in ("positive", "zero", "negative"):
                probabilities = [manifest.observation_probability("calibration", condition, prior, reliability, None, a) for a in (0, 1)]
                self.assertAlmostEqual(sum(probabilities), 1)
            independent = [manifest.observation_probability("core", "independent", prior, reliability, s, a) for s, a in itertools.product((0, 1), repeat=2)]
            same = [manifest.observation_probability("core", "same", prior, reliability, s, s) for s in (0, 1)]
            self.assertAlmostEqual(sum(independent), 1)
            self.assertAlmostEqual(sum(same), 1)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = manifest.build_rows()

    def test_counts_uniqueness_and_finite_oracles(self):
        self.assertEqual(len(self.rows), 612)
        self.assertEqual(collections.Counter(row["stage"] for row in self.rows), {"calibration": 108, "core": 504})
        self.assertEqual(len({row["prompt"] for row in self.rows}), 612)
        self.assertEqual(len({row["id"] for row in self.rows}), 612)
        self.assertEqual(len({row["world_id"] for row in self.rows}), 18)
        self.assertEqual(len({row["family_id"] for row in self.rows}), 6)
        self.assertTrue(all(0 < row["oracle"] < 1 for row in self.rows))

    def test_self_other_are_exact_counterparts(self):
        pairs = collections.defaultdict(dict)
        for row in self.rows:
            if row["attribution"] in ("self", "other"):
                key = tuple(row[field] for field in ("world_id", "template_id", "condition", "signal", "answer"))
                pairs[key][row["attribution"]] = row
        self.assertEqual(len(pairs), 216)
        for pair in pairs.values():
            self.assertEqual(set(pair), {"self", "other"})
            self.assertEqual(pair["self"]["oracle"], pair["other"]["oracle"])
            self.assertEqual(pair["self"]["prompt"], pair["other"]["prompt"].replace("another assistant", "you"))

    def test_readout_contains_no_numeric_or_json_example(self):
        self.assertIn("p_h1", manifest.OUTPUT_INSTRUCTION)
        self.assertIn("probability you compute", manifest.OUTPUT_INSTRUCTION)
        self.assertNotIn(".123", manifest.OUTPUT_INSTRUCTION)
        self.assertNotIn("{", manifest.OUTPUT_INSTRUCTION)
        self.assertNotIn("}", manifest.OUTPUT_INSTRUCTION)
        self.assertNotRegex(manifest.OUTPUT_INSTRUCTION, r"\d+\.\d+")
        self.assertEqual(manifest.GENERATOR_VERSION, "i16-dependency-attribution-v2-readout-no-example")
        self.assertTrue(all(row["prompt"].endswith(manifest.OUTPUT_INSTRUCTION) for row in self.rows))

    def test_only_legal_same_and_all_independent_values(self):
        independent_groups = collections.defaultdict(set)
        for row in self.rows:
            if row["condition"] == "same":
                self.assertEqual(row["signal"], row["answer"])
            if row["condition"] == "independent":
                key = tuple(row[field] for field in ("world_id", "template_id", "attribution"))
                independent_groups[key].add((row["signal"], row["answer"]))
        for values in independent_groups.values():
            self.assertEqual(values, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_determinism_and_seed_changes_order_only(self):
        self.assertEqual(self.rows, manifest.build_rows())
        another_order = manifest.build_rows(seed=1602)
        self.assertNotEqual([r["id"] for r in self.rows], [r["id"] for r in another_order])
        self.assertEqual({r["id"] for r in self.rows}, {r["id"] for r in another_order})

    def test_duplicate_grid_rejected(self):
        with self.assertRaises(ValueError):
            manifest.build_rows(priors=(.5, .5))

    def test_formal_default_prompts_disjoint_from_pilot(self):
        formal = manifest.build_rows(split="formal")
        self.assertEqual(len(formal), 5440)
        self.assertFalse({r["prompt"] for r in self.rows} & {r["prompt"] for r in formal})

    def test_immutable_write_idempotence_and_conflict(self):
        with tempfile.TemporaryDirectory(prefix="i16-manifest-test-") as directory:
            path = Path(directory) / "manifest.jsonl"
            self.assertTrue(manifest.immutable_write(path, b"original\n"))
            self.assertFalse(manifest.immutable_write(path, b"original\n"))
            with self.assertRaises(FileExistsError):
                manifest.immutable_write(path, b"replacement\n")
            self.assertEqual(path.read_bytes(), b"original\n")
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["manifest.jsonl"])


if __name__ == "__main__":
    unittest.main()
