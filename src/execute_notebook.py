"""Execute a Jupyter notebook end-to-end and verify it (Stage 7 gate).

Runs every cell with nbclient (live, cwd = notebook's project root so relative paths
and .env resolve), writes executed outputs back into the .ipynb, and reports:
  - any cell that errored
  - any cell with more than one rich output (violates the one-output-per-cell rule)

Usage:
    python execute_notebook.py path/to/analysis.ipynb [--timeout 900]

Requires: nbclient, nbformat, ipykernel  (e.g. `uv add --dev nbclient nbformat ipykernel`).
"""
import argparse
import os
import sys

import nbformat
from nbclient import NotebookClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    path = os.path.abspath(args.notebook)
    workdir = os.path.dirname(path)
    nb = nbformat.read(path, as_version=4)

    client = NotebookClient(
        nb, timeout=args.timeout, kernel_name="python3",
        resources={"metadata": {"path": workdir}},
    )
    try:
        client.execute()
    except Exception as e:  # noqa: BLE001 - surface execution failure to the caller
        print(f"EXECUTION FAILED: {type(e).__name__}: {e}")
        nbformat.write(nb, path)  # persist partial outputs for debugging
        sys.exit(1)

    nbformat.write(nb, path)

    errors, multi = 0, 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        rich = 0
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors += 1
                print(f"  ERROR cell {i}: {out.get('ename')}: {out.get('evalue')}")
            if out.get("output_type") in ("display_data", "execute_result"):
                rich += 1
        if rich > 1:
            multi += 1
            print(f"  >1 output cell {i}: {cell.source.splitlines()[0][:60]!r} ({rich})")

    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"executed {code_cells} code cells | errors: {errors} | "
          f"cells with >1 rich output: {multi}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
