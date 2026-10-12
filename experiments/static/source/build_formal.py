#!/usr/bin/env python3
"""Build the I16 formal candidate; creating a file does not freeze the protocol.

This candidate awaits development-readout review. It introduces an explicit vs.
minimal reminder contrast while preserving the factual attribution mechanism.
Controls are shared observations, never duplicated model calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from build_manifest import build_rows, canonical_json, digest, immutable_write


GENERATOR_VERSION = "i16-dependency-attribution-formal-candidate-v1"
PRIORS = (0.15, 0.30, 0.45, 0.55, 0.70, 0.85)
RELIABILITIES = (0.60, 0.70, 0.75, 0.85, 0.95)
REMINDER_SENTENCES = {
    "direct_v1": (
        "This attribution was assigned independently of H and the readings and "
        "does not change the reporting rule or provide any other evidence.",
        "This attribution was assigned independently of H and the readings.",
    ),
    "record_v1": (
        "The author attribution is assigned independently of the state and "
        "readings; it adds no information and leaves the reporting rule fixed.",
        "The author attribution is assigned independently of the state and readings.",
    ),
}


def with_reminder(source: dict, reminder: str) -> dict:
    """Preserve the science and change only the designated reminder sentence."""
    row = {key: value for key, value in source.items() if key != "id"}
    is_core_attribution = row["condition"] in ("same", "independent")
    if is_core_attribution:
        if reminder not in ("explicit", "minimal"):
            raise ValueError("Core attribution conditions require an explicit or minimal reminder")
        full, minimal = REMINDER_SENTENCES[row["template_id"]]
        if row["prompt"].count(full) != 1:
            raise ValueError("Source reminder sentence changed; review the candidate before regenerating")
        if reminder == "minimal":
            row["prompt"] = row["prompt"].replace(full, minimal, 1)
    elif reminder != "shared":
        raise ValueError("Calibration and baseline must be single shared observations")
    row["reminder"] = reminder
    row["generator_version"] = GENERATOR_VERSION
    row["id"] = digest(row)
    return row


def build_formal_rows(seed: int = 1602) -> list[dict]:
    sources = build_rows(split="formal", seed=0, priors=PRIORS, reliabilities=RELIABILITIES)
    rows = []
    for source in sorted(sources, key=lambda row: row["id"]):
        reminders = ("explicit", "minimal") if source["condition"] in ("same", "independent") else ("shared",)
        rows.extend(with_reminder(source, reminder) for reminder in reminders)
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("Candidate contains duplicate prompts")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Candidate contains duplicate identifiers")
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1602, help="Randomizes order only")
    args = parser.parse_args()
    rows = build_formal_rows(args.seed)
    content = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    created = immutable_write(args.output, content)
    print(json.dumps({
        "path": str(args.output.resolve()), "created": created,
        "protocol_status": "candidate_not_frozen", "rows": len(rows),
        "stages": dict(Counter(row["stage"] for row in rows)),
        "conditions": dict(Counter(row["condition"] for row in rows)),
        "reminders": dict(Counter(row["reminder"] for row in rows)),
        "worlds": len({row["world_id"] for row in rows}),
        "families": len({row["family_id"] for row in rows}),
        "seed": args.seed, "split": "formal",
        "sha256": hashlib.sha256(content).hexdigest(),
        "generator_version": GENERATOR_VERSION,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
