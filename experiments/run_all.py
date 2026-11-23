"""Run the complete experiment pipeline and print a one-screen summary.

    python experiments/run_all.py

Each experiment also runs on its own, e.g. ``python experiments/experiment_05_reproduce_paper.py``.
All outputs go to ``results/`` (figures, tables, raw JSON/CSV, logs, configs).
"""

from __future__ import annotations

import _common as C  # noqa: I001

import importlib
import sys
import time
import traceback

EXPERIMENTS = [
    ("experiment_01_forward_validation", "Forward solver validation"),
    ("experiment_02_noiseless_reconstruction", "Noise-free reconstruction (supplementary)"),
    ("experiment_03_noise_sensitivity", "Noise sensitivity"),
    ("experiment_04_regularization_parameter", "Regularisation parameter (supplementary)"),
    ("experiment_05_reproduce_paper", "Reproduction of Tables 1-2 and Figures 1-2"),
]


def main() -> int:
    results, failures = {}, []
    t0 = time.perf_counter()
    for module_name, title in EXPERIMENTS:
        print(f"\n{'=' * 78}\n>>> {title}  ({module_name}.py)\n{'=' * 78}", flush=True)
        try:
            results[module_name] = importlib.import_module(module_name).run()
        except Exception:  # noqa: BLE001 - report and continue, but never silently
            traceback.print_exc()
            failures.append(module_name)

    try:  # README hero image
        import subprocess
        subprocess.run([sys.executable, str(C.ROOT / "demo.py"), "--snapshot"], check=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failures.append("demo snapshot")

    print(f"\n{'=' * 78}\nSUMMARY  ({time.perf_counter() - t0:.1f} s)\n{'=' * 78}")
    r1 = results.get("experiment_01_forward_validation")
    if r1:
        print(f"[Exp 1] spectral forward solve: max |u(T) - g| = {r1['spectral_max_abs_error']:.2e}")
        print(f"        Crank-Nicolson observed orders: {[round(o, 2) for o in r1['crank_nicolson']['observed_order'][1:]]}")
    r2 = results.get("experiment_02_noiseless_reconstruction")
    if r2:
        b = r2["best_cutoff"]["0.0"]
        print(f"[Exp 2] noise-free, t=0: best M = {b['M']:g} (rel. L2 {b['relative_l2']:.3f}); "
              f"M = 80 gives rel. L2 {r2['M80_relative_l2_t0']:.1e} (rounding x e^80)")
    r5 = results.get("experiment_05_reproduce_paper")
    if r5:
        print(f"[Exp 5] Table 1: max relative deviation from the printed values {r5['table1_max_rel_dev']:.1e}")
        print(f"        Table 2: median |rel. diff| {r5['table2_median_abs_rel_diff']:.1e}, "
              f"max {r5['table2_max_abs_rel_diff']:.1e} (119 entries; typo cell excluded)")
        print(f"        typo cell eps=1e-5, M3, t=0.1: ours {r5['typo_cell']['ours']:.4e} vs printed {r5['typo_cell']['printed']:.4e}")
    print(f"\nFigures: {C.FIGURES_DIR}\nTables:  {C.TABLES_DIR}")
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll experiments completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
