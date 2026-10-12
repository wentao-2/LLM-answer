"""Refresh supplementary table rows from the included frozen-source extract.

Run from any directory with Python 3. No external packages are required.
The extract preserves numeric values and source hashes. This script changes
table rows only, retaining the manuscript's captions, headers, and prose.
It neither reruns inference nor recalculates statistical estimates.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value):
    if value is None:
        return "NA"
    if abs(value) < 1e-12:
        return "$0$"
    if abs(value) < 1e-4:
        mantissa, exponent = f"{value:.2e}".split("e")
        return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
    return "$" + f"{value:.6f}" + "$"


def interval(values):
    lower, upper = values
    if abs(lower - upper) < 1e-12:
        return number(lower)
    return "$[" + number(lower)[1:-1] + ", " + number(upper)[1:-1] + "]$"


def row(cells):
    rendered = []
    for cell in cells:
        if isinstance(cell, dict):
            kind = cell["format"]
            value = cell["value"]
            if kind == "number":
                rendered.append(number(value))
            elif kind == "bounds":
                rendered.append(interval(value))
            elif kind == "descriptive_interval":
                rendered.append(f"$[{value[0]:.6f}, {value[1]:.6f}]$")
            elif kind == "percent_pair":
                rendered.append(f"{100 * value[0]:.2f} / {100 * value[1]:.2f}")
            else:
                raise ValueError(f"Unknown cell format: {kind}")
        else:
            rendered.append(str(cell))
    return " & ".join(rendered) + r" \\"


def main():
    data = json.loads((ROOT / "data/result_table_sources.json").read_text(encoding="utf-8"))
    path = ROOT / "sections/result_appendix.tex"
    manuscript = path.read_text(encoding="utf-8")
    for table in data["tables"]:
        label = table["label"]
        pattern = re.compile(
            r"(\\begin\{table\*\}.*?\\begin\{tabular\}.*?\\toprule\n"
            r".*?\\midrule\n)(.*?)(\\bottomrule.*?\\label\{" + re.escape(label)
            + r"\}.*?\\end\{table\*\})", re.S
        )
        # Start with the table carrying this label, rather than a preceding float.
        floats = list(re.finditer(r"\\begin\{table\*\}.*?\\end\{table\*\}", manuscript, re.S))
        targets = [m for m in floats if r"\label{" + label + "}" in m.group()]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one table for {label}")
        target = targets[0]
        block = target.group()
        match = pattern.fullmatch(block)
        if match is None:
            raise ValueError(f"Unrecognized table structure: {label}")
        body = "\n".join(item if isinstance(item, str) else row(item)
                         for item in table["rows"]) + "\n"
        replacement = match.group(1) + body + match.group(3)
        manuscript = manuscript[:target.start()] + replacement + manuscript[target.end():]
    path.write_text(manuscript, encoding="utf-8")
    print(f"Refreshed {len(data['tables'])} tables in {path}")


if __name__ == "__main__":
    main()
