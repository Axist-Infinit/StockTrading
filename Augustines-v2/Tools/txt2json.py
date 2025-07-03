#!/usr/bin/env python3
"""
txt2json_universal.py  –  Convert any FinViz‑style table to JSON

Usage
-----
    python txt2json_universal.py  INPUT.txt  OUTPUT.json
"""

import json
import re
import sys
from pathlib import Path


def smart_split(line: str) -> list[str]:
    """Prefer tab split; otherwise split on runs of ≥2 spaces."""
    return line.rstrip("\n").split("\t") if "\t" in line else re.split(r"\s{2,}", line.strip())


def normalise_header(raw: list[str]) -> list[str]:
    """Make header names JSON‑key friendly."""
    cleaned = []
    for h in raw:
        h = h.strip().replace(" ", "_")
        h = re.sub(r"[^\w]", "", h)      # keep A‑Z a‑z 0‑9 _
        h = re.sub(r"__+", "_", h)       # collapse repeats
        cleaned.append(h)
    return cleaned


def parse_table(path: Path) -> list[dict]:
    rows, header = [], None
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            cells = smart_split(ln)
            if header is None:
                header = normalise_header(cells)
                continue
            # defensive pad / trim
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(dict(zip(header, cells)))
    return rows


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    data = parse_table(src)
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✔  {len(data)} records written → {dst}")


if __name__ == "__main__":
    main()

