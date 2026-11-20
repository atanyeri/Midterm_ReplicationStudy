#!/usr/bin/env python3
"""The one command: run replication.ipynb end to end, then print the headline table.

    python reproduce.py

The notebook runs top to bottom in a fresh kernel, from the repository root, with
the Python that runs this script (so the packages are the ones requirements.txt
installed). The executed copy, with its outputs, is written to
output/executed/replication.ipynb. The run stops at the first cell that fails and
prints that cell's traceback.

The data is fetched first if data/hansen_dwi.dta is missing.
The headline table is output/headline_table.csv, which the notebook writes.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pandas as pd
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "replication.ipynb"
OUTPUT = ROOT / "output"
DATA = ROOT / "data" / "hansen_dwi.dta"
HEADLINE = OUTPUT / "headline_table.csv"
CELL_TIMEOUT = 1800          # seconds per cell; raise it if one estimation takes longer


class ThisPythonKernel(KernelManager):
    """A kernel that runs on this interpreter, not on whatever 'python3' kernel the machine has registered."""

    @property
    def kernel_spec(self) -> KernelSpec:
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name=f"Python ({sys.executable})", language="python")


def fetch_data() -> None:
    """hansen_dwi.dta is not committed; download_data.py fetches and checksums it."""
    if DATA.exists():
        return
    print(f"{DATA.relative_to(ROOT)} missing; fetching it.", flush=True)
    if subprocess.run([sys.executable, str(ROOT / "download_data.py")], cwd=ROOT).returncode != 0:
        sys.exit("Could not fetch the data. See download_data.py for the URL to save by hand.")


def run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_manager_class=ThisPythonKernel,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        out = OUTPUT / "executed" / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out)


def main() -> None:
    if not NOTEBOOK.exists():
        sys.exit(f"{NOTEBOOK.name} not found. Run this from the repository root.")
    OUTPUT.mkdir(exist_ok=True)
    fetch_data()

    print(f"running {NOTEBOOK.name} ...", flush=True)
    start = time.time()
    try:
        run(NOTEBOOK)
    except CellExecutionError as err:
        print(f"\nFAILED in {NOTEBOOK.name}. The executed copy is in output/executed/.\n")
        print(err)
        sys.exit(1)
    print(f"  done in {time.time() - start:.0f} s\n")

    if not HEADLINE.exists():
        print("No headline table yet: output/headline_table.csv does not exist.\n"
              "The replication table cell in replication.ipynb should end with\n"
              '    table.to_csv("output/headline_table.csv")')
        return
    table = pd.read_csv(HEADLINE, index_col=0)
    if table.empty:
        print("output/headline_table.csv has no rows yet.")
        return
    print("Headline table (output/headline_table.csv)\n")
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(table.to_string())


if __name__ == "__main__":
    main()