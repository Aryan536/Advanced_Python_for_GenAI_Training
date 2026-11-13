"""Run all project steps (p00 ... p14) in order, sharing variables like a notebook.

Usage (from the repository root):  python src/module6_project/run_all.py
"""
import pathlib

here = pathlib.Path(__file__).parent
namespace = {"__name__": "__main__"}
for step in sorted(here.glob("p[0-9][0-9]_*.py")):
    print(f"\n{'=' * 20} {step.name} {'=' * 20}")
    exec(compile(step.read_text(), str(step), "exec"), namespace)
