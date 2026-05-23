r"""
Lance plusieurs simulations V2X avec differents eps0 et trace le taux de succes.

Exemples :
    .venv\Scripts\python.exe simulation\sweep_eps0.py
    .venv\Scripts\python.exe simulation\sweep_eps0.py 0 0.05 0.1 0.2 0.4
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
SIM_SCRIPT = ROOT_DIR / "simulation" / "run_simulation.py"
OUT_DIR = ROOT_DIR / "resultats"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_EPS0_VALUES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0]
SUCCESS_RE = re.compile(r"Taux de succes MAB\s+:\s+([0-9.]+)%")


def run_one_eps0(eps0: float) -> float:
    env = os.environ.copy()
    env["MAB_EPS0"] = str(eps0)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        [sys.executable, str(SIM_SCRIPT)],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        print(output)
        raise RuntimeError(f"Simulation echouee pour eps0={eps0}")

    match = SUCCESS_RE.search(output)
    if not match:
        print(output)
        raise RuntimeError(f"Taux de succes introuvable pour eps0={eps0}")

    return float(match.group(1)) / 100.0


def plot_results(rows):
    eps0_values = [row["eps0"] for row in rows]
    success_rates = [row["success_rate"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(eps0_values, success_rates, marker="o", linewidth=2)
    plt.xlabel("eps0")
    plt.ylabel("Taux de reussite")
    plt.ylim(0.0, 1.05)
    plt.title("Impact de eps0 sur le taux de reussite MAB eps-greedy")
    plt.grid(True, alpha=0.3)

    plot_path = OUT_DIR / "sweep_eps0_taux_reussite.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def write_csv(rows):
    csv_path = OUT_DIR / "sweep_eps0_taux_reussite.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["eps0", "success_rate", "success_percent"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "eps0": row["eps0"],
                "success_rate": row["success_rate"],
                "success_percent": row["success_rate"] * 100.0,
            })
    return csv_path


def parse_eps0_values(args):
    if not args:
        return DEFAULT_EPS0_VALUES
    return [float(arg.replace(",", ".")) for arg in args]


def main():
    eps0_values = parse_eps0_values(sys.argv[1:])
    rows = []

    print("=" * 65)
    print("  Sweep eps0 - MAB eps-greedy")
    print("=" * 65)

    for eps0 in eps0_values:
        print(f"  Simulation eps0={eps0:.3f} ...", flush=True)
        success_rate = run_one_eps0(eps0)
        rows.append({"eps0": eps0, "success_rate": success_rate})
        print(f"    taux de reussite = {success_rate:.1%}")

    csv_path = write_csv(rows)
    plot_path = plot_results(rows)

    print()
    print(f"  CSV sauvegarde   : {csv_path}")
    print(f"  Image sauvegardee: {plot_path}")


if __name__ == "__main__":
    main()
