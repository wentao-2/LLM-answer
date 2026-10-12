#!/usr/bin/env python3
"""Build immutable, deterministic I16 synthetic prompt manifests (stdlib only)."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile


GENERATOR_VERSION = "i16-dependency-attribution-v2-readout-no-example"
PILOT_PRIORS = (0.25, 0.5, 0.75)
PILOT_RELIABILITIES = (0.65, 0.8, 0.9)
# This larger grid is a candidate, not an automatically approved final protocol.
# Its numerical settings are disjoint from the pilot grid.
FORMAL_PRIORS = (0.1, 0.15, 0.2, 0.35, 0.4, 0.6, 0.65, 0.8, 0.85, 0.9)
FORMAL_RELIABILITIES = (0.6, 0.625, 0.675, 0.7, 0.75, 0.825, 0.85, 0.95)
CONTEXTS = {
    "sensor": (
        "In this hypothetical case, a sealed device has an unknown binary internal state H, either 0 or 1. "
        "A sensor takes one reading of that state."
    ),
    "laboratory": (
        "In this hypothetical case, a sealed sample has an unknown binary property H, either 0 or 1. "
        "A laboratory assay takes one reading of that property."
    ),
}
OUTPUT_INSTRUCTION = (
    "Return exactly one JSON object containing only the key p_h1. Its value must "
    "be the probability you compute that H=1 from the evidence stated above. "
    "Use a numeric decimal value between zero and one, inclusive, not a "
    "percentage. Do not include reasoning, additional keys, or any other text."
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: object, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def number(value: float) -> str:
    return format(value, ".12g")


def probability_h1(prior: float, reliability: float, signals: tuple[int, ...]) -> float:
    """Bayes posterior for conditionally independent, symmetric binary readings."""
    if not 0 < prior < 1 or not 0.5 < reliability < 1:
        raise ValueError("Require 0 < prior < 1 and 0.5 < reliability < 1")
    if any(s not in (0, 1) for s in signals):
        raise ValueError("Signals must be binary")
    p1, p0 = prior, 1 - prior
    for signal in signals:
        p1 *= reliability if signal else 1 - reliability
        p0 *= 1 - reliability if signal else reliability
    return p1 / (p1 + p0)


def oracle_posterior(stage: str, condition: str, prior: float, reliability: float,
                     signal: int | None, answer: int | None) -> float:
    """Compute the posterior from the declared stochastic mechanism, not text."""
    if stage == "calibration":
        if signal is not None or answer not in (0, 1):
            raise ValueError("Calibration exposes A only")
        if condition == "positive":
            return probability_h1(prior, reliability, (answer,))
        if condition == "negative":
            return probability_h1(prior, reliability, (1 - answer,))
        if condition == "zero":
            probability_h1(prior, reliability, ())  # Validate parameters.
            return prior
        raise ValueError("Unknown calibration condition")
    if stage != "core" or signal not in (0, 1):
        raise ValueError("Core must expose binary S")
    if condition == "baseline" and answer is None:
        return probability_h1(prior, reliability, (signal,))
    if condition == "same" and answer == signal:
        return probability_h1(prior, reliability, (signal,))
    if condition == "independent" and answer in (0, 1):
        return probability_h1(prior, reliability, (signal, answer))
    raise ValueError("Illegal condition/evidence combination")


def observation_probability(stage: str, condition: str, prior: float, reliability: float,
                            signal: int | None, answer: int | None) -> float:
    """P(exposed evidence), useful if a later analysis chooses natural weighting."""
    oracle_posterior(stage, condition, prior, reliability, signal, answer)
    if stage == "calibration":
        if condition == "zero":
            return 0.5
        signals = (answer if condition == "positive" else 1 - answer,)
    else:
        signals = (signal, answer) if condition == "independent" else (signal,)
    p1, p0 = prior, 1 - prior
    for value in signals:
        p1 *= reliability if value else 1 - reliability
        p0 *= 1 - reliability if value else reliability
    return p1 + p0


def prompt_for(stage: str, condition: str, attribution: str, prior: float,
               reliability: float, signal: int | None, answer: int | None,
               template_id: str, context_id: str) -> str:
    p, r = number(prior), number(reliability)
    introduction = CONTEXTS[context_id]
    if template_id == "direct_v1":
        base = (
            f"{introduction}\nBefore any readings, P(H=1)={p}. "
            f"The first reading S is binary: P(S=1|H=1)={r} and P(S=0|H=0)={r}. "
            "All complementary probabilities are determined by these values. "
            "The reporting procedure described below was fixed before H or any readings "
            "were generated. This case was not selected, discarded, or repeated based "
            "on H, S, A, or agreement between readings.\n"
        )
        if stage == "calibration":
            mechanisms = {
                "positive": "The recorded answer A is produced by the fixed rule A=S.",
                "zero": (
                    "The recorded answer A is the result of a fair binary coin, "
                    "independent of H and S: P(A=1|H,S)=0.5."
                ),
                "negative": "The recorded answer A is produced by the fixed rule A=1-S.",
            }
            evidence = f"{mechanisms[condition]} The original S is hidden. You observe only A={answer}."
        elif condition == "baseline":
            evidence = f"You observe S={signal}. No earlier answer or other reading is available."
        else:
            if condition == "same":
                mechanism = (
                    "An earlier answer A was recorded by faithfully copying the same first "
                    "reading S, using the fixed rule A=S. No second reading was taken."
                )
            else:
                mechanism = (
                    "A second reading T was generated independently of S conditional on H: "
                    f"P(T=1|H=1)={r} and P(T=0|H=0)={r}. An earlier answer A was recorded "
                    "by faithfully copying T, using the fixed rule A=T."
                )
            owner = "you" if attribution == "self" else "another assistant"
            evidence = (
                f"{mechanism}\nThe record attributes this earlier answer to {owner}. "
                "This attribution was assigned independently of H and the readings and "
                "does not change the reporting rule or provide any other evidence.\n"
                f"You observe S={signal} and the recorded earlier answer A={answer}."
            )
    elif template_id == "record_v1":
        base = (
            f"{introduction}\nConsider the following complete case specification.\n"
            f"Prior chance of state 1: {p}.\nFirst binary reading: S. Its chance of matching "
            f"H is {r} for H=0 and also {r} for H=1.\nThe reporting rule is chosen "
            "in advance, independently of the state and readings. There is no selection "
            "or stopping rule based on the state, observed values, or whether the "
            "readings agree.\n"
        )
        if stage == "calibration":
            mechanisms = {
                "positive": "Reporting rule: copy S unchanged into A (A=S).",
                "zero": "Reporting rule: set A with an independent fair coin, unrelated to H or S.",
                "negative": "Reporting rule: flip S into A (A=1-S).",
            }
            evidence = f"{mechanisms[condition]}\nAvailable record: A={answer}. S is unavailable."
        elif condition == "baseline":
            evidence = f"Available record: S={signal}. There is no earlier answer and no additional reading."
        else:
            if condition == "same":
                mechanism = (
                    "Earlier-answer rule: copy the first reading S unchanged into A "
                    "(A=S). There was no second reading."
                )
            else:
                mechanism = (
                    f"A second binary reading T matches H with probability {r} in each "
                    "state. Conditional on H, T and S are independent. Earlier-answer "
                    "rule: copy T unchanged into A (A=T)."
                )
            owner = "you" if attribution == "self" else "another assistant"
            evidence = (
                f"{mechanism}\nAuthor attributed to the earlier answer: {owner}. "
                "The author attribution is assigned independently of the state and "
                "readings; it adds no information and leaves the reporting rule fixed.\n"
                f"Available record: S={signal}; earlier answer A={answer}."
            )
    else:
        raise ValueError(f"Unknown template: {template_id}")
    return f"{base}{evidence}\n\n{OUTPUT_INSTRUCTION}"


def build_rows(split: str = "pilot", seed: int = 1601,
               priors: tuple[float, ...] | None = None,
               reliabilities: tuple[float, ...] | None = None) -> list[dict]:
    if split not in ("pilot", "formal"):
        raise ValueError("split must be pilot or formal")
    if priors is None:
        priors = PILOT_PRIORS if split == "pilot" else FORMAL_PRIORS
    if reliabilities is None:
        reliabilities = PILOT_RELIABILITIES if split == "pilot" else FORMAL_RELIABILITIES
    if len(set(priors)) != len(priors) or len(set(reliabilities)) != len(reliabilities):
        raise ValueError("Duplicate numerical settings would create duplicate prompts")
    rows = []
    for prior, reliability in itertools.product(priors, reliabilities):
        probability_h1(prior, reliability, ())
        for stage in ("calibration", "core"):
            contexts = ("sensor",) if stage == "calibration" else tuple(CONTEXTS)
            for context_id, template_id in itertools.product(contexts, ("direct_v1", "record_v1")):
                conditions = ("positive", "zero", "negative") if stage == "calibration" else ("same", "independent", "baseline")
                for condition in conditions:
                    if stage == "calibration":
                        evidence_values = [(None, a) for a in (0, 1)]
                    elif condition == "baseline":
                        evidence_values = [(s, None) for s in (0, 1)]
                    elif condition == "same":
                        evidence_values = [(s, s) for s in (0, 1)]
                    else:
                        evidence_values = list(itertools.product((0, 1), repeat=2))
                    attributions = ("self", "other") if condition in ("same", "independent") else ("none",)
                    for (signal, answer), attribution in itertools.product(evidence_values, attributions):
                        world = {"prior": prior, "reliability": reliability, "context_id": context_id}
                        family = {"prior_pair": round(min(prior, 1 - prior), 12), "reliability": reliability}
                        row = {
                            "generator_version": GENERATOR_VERSION,
                            "world_id": digest({"split": split, **world}),
                            "family_id": digest({"split": split, **family}),
                            "stage": stage, "template_id": template_id,
                            "context_id": context_id, "condition": condition,
                            "attribution": attribution, "prior": prior,
                            "reliability": reliability, "signal": signal,
                            "answer": answer, "split": split,
                            "oracle": oracle_posterior(stage, condition, prior, reliability, signal, answer),
                            "evidence_probability": observation_probability(stage, condition, prior, reliability, signal, answer),
                            "prompt": prompt_for(stage, condition, attribution, prior, reliability,
                                                 signal, answer, template_id, context_id),
                        }
                        row["id"] = digest(row)
                        rows.append(row)
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prompts detected")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate identifiers detected")
    random.Random(seed).shuffle(rows)
    return rows


def immutable_write(path: Path, content: bytes) -> bool:
    """Publish a fully written file without ever replacing an existing target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace differing immutable manifest: {path}")
        return False
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"Refusing concurrent differing manifest: {path}")
            return False
        return True
    finally:
        os.unlink(temporary)


def parse_grid(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated probabilities") from error
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("Grid must contain finite probabilities")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--seed", type=int, default=1601, help="Order only; does not create new independent samples")
    parser.add_argument("--priors", type=parse_grid)
    parser.add_argument("--reliabilities", type=parse_grid)
    args = parser.parse_args()
    rows = build_rows(args.split, args.seed, args.priors, args.reliabilities)
    content = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    created = immutable_write(args.output, content)
    print(json.dumps({
        "path": str(args.output.resolve()), "created": created, "rows": len(rows),
        "stages": dict(Counter(row["stage"] for row in rows)),
        "worlds": len({row["world_id"] for row in rows}),
        "families": len({row["family_id"] for row in rows}),
        "seed": args.seed, "split": args.split,
        "sha256": hashlib.sha256(content).hexdigest(),
        "generator_version": GENERATOR_VERSION,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
