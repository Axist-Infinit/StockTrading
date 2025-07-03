#!/usr/bin/env python3
"""
txt2json.py  –  Convert FinViz‑style industry tables to JSON

Usage
-----
    python txt2json.py  INPUT.txt  OUTPUT.json

• INPUT.txt  – file that contains the raw table (the block you pasted).
• OUTPUT.json – file to write the JSON array to.

The script:
1. Reads the first non‑blank line as the header.
2. Accepts either real tab characters or runs of ≥2 spaces as separators.
3. Normalises the header names (removes full stops, turns spaces into underscores).
4. Produces a list of dicts and writes it prettified (indent=2) to OUTPUT.json.
"""

import csv
import json
import re
import sys
from pathlib import Path


def smart_split(line: str) -> list[str]:
    """Split on tab if present, otherwise on 2+ consecutive spaces."""
    return line.rstrip("\n").split("\t") if "\t" in line else re.split(r"\s{2,}", line.strip())


def normalise_header(raw_header: list[str]) -> list[str]:
    out = []
    for h in raw_header:
        h = h.replace(".", "")          # 'No.' ➜ 'No'
        h = h.replace(" ", "_")         # 'Perf Week' ➜ 'Perf_Week'
        out.append(h)
    return out


def read_table(path: Path) -> list[dict]:
    records: list[dict] = []
    header: list[str] | None = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # skip blank lines

            cells = smart_split(line)
            if header is None:
                header = normalise_header(cells)
                continue

            # Pad short rows (defensive)
            cells += [""] * (len(header) - len(cells))
            record = dict(zip(header, cells))
            records.append(record)

    return records


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    in_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])

    data = read_table(in_file)
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✔  {len(data)} records written to {out_file}")


if __name__ == "__main__":
    main()

