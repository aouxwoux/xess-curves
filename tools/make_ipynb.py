"""Convert the percent-cell source notebook to a plain .ipynb file."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tess_xgboost_exoplanet_prototype.py"
TARGET = ROOT / "tess_xgboost_exoplanet_prototype.ipynb"


def flush_cell(cells, cell_type, lines):
    if cell_type is None:
        return
    if cell_type == "markdown":
        source = []
        for line in lines:
            if line.startswith("# "):
                source.append(line[2:])
            elif line.rstrip() == "#":
                source.append("\n")
            elif line.startswith("#"):
                source.append(line[1:].lstrip())
            else:
                source.append(line)
    else:
        source = lines

    cells.append(
        {
            "cell_type": cell_type,
            "metadata": {},
            "source": source,
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        }
    )


def main():
    cells = []
    cell_type = None
    lines = []

    for line in SOURCE.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush_cell(cells, cell_type, lines)
            cell_type = "markdown" if "[markdown]" in stripped else "code"
            lines = []
            continue
        lines.append(line)

    flush_cell(cells, cell_type, lines)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    TARGET.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
