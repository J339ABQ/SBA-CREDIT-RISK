"""Build + execute the notebooks from percent-format sources in scripts/notebooks/*.py.

    python scripts/build_notebooks.py            # all
    python scripts/build_notebooks.py 01_eda     # one
Cells are delimited by '# %%' (code) and '# %% [markdown]' (markdown; lines prefixed with '# ').
Executed notebooks (with outputs) are written to notebooks/.
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "notebooks"
OUT = ROOT / "notebooks"


def parse(path: Path):
    cells, mode, buf = [], None, []

    def flush():
        if mode is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if mode == "md":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in path.read_text().split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); mode, buf = "md", []
        elif line.startswith("# %%"):
            flush(); mode, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def build(name: str):
    nb = nbformat.v4.new_notebook(cells=parse(SRC / f"{name}.py"))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    for c in nb.cells:  # drop volatile execution metadata so re-runs diff cleanly
        c.metadata.pop("execution", None)
    OUT.mkdir(exist_ok=True)
    nbformat.write(nb, OUT / f"{name}.ipynb")
    print("built", name)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in SRC.glob("*.py"))
    for n in names:
        build(n)
