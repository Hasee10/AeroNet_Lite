"""
Execute AeroNet_Lite.ipynb, save all cell text outputs to log.txt,
and confirm PNG figures are saved in report/figures/.

Usage:
    cd notebooks
    python run_notebook.py
"""

import subprocess
import json
import os
import sys

NOTEBOOK    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AeroNet_Lite.ipynb")
LOG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures"))


def execute_notebook():
    print("Executing notebook (this may take ~30 seconds)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=180",
            NOTEBOOK,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: nbconvert failed.")
        print(result.stderr)
        sys.exit(1)
    print("Notebook executed successfully.")


def extract_outputs_to_log():
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    lines = [
        "AeroNet Lite — Notebook Output Log",
        "=" * 60,
        "",
    ]

    code_cell_index = 0
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        code_cell_index += 1
        outputs = cell.get("outputs", [])

        text_parts = []
        for out in outputs:
            otype = out.get("output_type", "")
            if otype == "stream":
                text_parts.append("".join(out.get("text", [])))
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                plain = data.get("text/plain", [])
                if plain:
                    text_parts.append("".join(plain))

        if text_parts:
            lines.append(f"[Cell {code_cell_index}]")
            lines.append("".join(text_parts).rstrip())
            lines.append("-" * 40)
            lines.append("")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Log saved  → {LOG_PATH}")


def check_figures():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pngs = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    if pngs:
        print(f"Figures    → {FIGURES_DIR}")
        for name in sorted(pngs):
            print(f"             {name}")
    else:
        print(f"Figures dir created (no PNGs yet): {FIGURES_DIR}")


if __name__ == "__main__":
    execute_notebook()
    extract_outputs_to_log()
    check_figures()
