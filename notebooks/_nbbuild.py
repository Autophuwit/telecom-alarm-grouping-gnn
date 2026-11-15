"""Execute a list of notebook cells and write an .ipynb with their outputs (for machines without Jupyter).

    from _nbbuild import build
    build(CELLS, Path("notebooks/02_model_and_clustering.ipynb"))

CELLS: [("md", text) | ("code", source)]. Code runs in one namespace with the notebook's folder as the working
directory; stdout, matplotlib figures and the value of a trailing expression become outputs.
"""
import ast
import base64
import contextlib
import io
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def _run(src, ns):
    outputs, buf = [], io.StringIO()
    tree = ast.parse(src)
    last = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)
        val = eval(compile(last, "<cell>", "eval"), ns) if last else None
    if buf.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
    for num in plt.get_fignums():
        png = io.BytesIO()
        plt.figure(num).savefig(png, format="png", dpi=110, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": "<Figure>"}})
    plt.close("all")
    return outputs, val


def build(cells, out: Path, echo=True):
    out = Path(out).resolve()
    here = os.getcwd()
    os.chdir(out.parent)
    try:
        ns, nb_cells, n = {"__name__": "__main__"}, [], 0
        for kind, src in cells:
            lines = src.strip("\n").splitlines(True)
            if kind == "md":
                nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
                continue
            n += 1
            outputs, val = _run(src, ns)
            if val is not None:
                data = {"text/plain": repr(val).splitlines(True)}
                if hasattr(val, "to_html"):
                    data["text/html"] = val.to_html().splitlines(True)
                outputs.append({"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": data})
            nb_cells.append({"cell_type": "code", "execution_count": n, "metadata": {}, "outputs": outputs, "source": lines})
            if echo:
                for o in outputs:
                    if o["output_type"] == "stream":
                        print("".join(o["text"]), end="")
    finally:
        os.chdir(here)
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
