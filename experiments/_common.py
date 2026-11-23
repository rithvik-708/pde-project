"""Shared plumbing for the experiment scripts (paths, logging, problem setup, I/O).

Every experiment script imports this module first. It makes ``src/`` importable, so the
scripts run from a fresh checkout without ``pip install -e .``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backward_heat import Grid, PaperExample, SeparableExponentialSource  # noqa: E402
from backward_heat.config import (  # noqa: E402
    DEFAULT_CONFIG, FIGURES_DIR, LOGS_DIR, RAW_DIR, TABLES_DIR, ExperimentConfig, ensure_result_dirs,
)


@dataclass(frozen=True)
class Setup:
    """Objects every experiment needs."""

    cfg: ExperimentConfig
    grid: Grid
    example: PaperExample
    source: SeparableExponentialSource
    log: logging.Logger

    @property
    def T(self) -> float:  # noqa: N802 - paper notation
        return self.example.final_time


def setup(name: str, cfg: ExperimentConfig = DEFAULT_CONFIG) -> Setup:
    """Create result dirs, configure logging (console + results/logs/<name>.log), save config."""
    ensure_result_dirs()
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS_DIR / f"{name}.log", "w", "utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.propagate = False
    cfg.save(RAW_DIR / f"{name}_config.json")
    grid = Grid.from_config(cfg.grid)
    example = PaperExample.from_config(cfg.problem)
    source = SeparableExponentialSource(example.f_spatial_coefficients(grid), example.time_rate())
    log.info("grid %dx%d nodes, T=%.2f, K=%d, seed=%d", grid.n + 1, grid.n + 1, example.final_time,
             example.n_terms, cfg.seed)
    return Setup(cfg, grid, example, source, log)


def save_table(df: pd.DataFrame, name: str, float_format: str = "%.4e") -> None:
    """Write a table as CSV (full precision) and Markdown (readable)."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    fmt = float_format.strip("%")
    floatfmt = tuple(".0f" if pd.api.types.is_integer_dtype(df[c]) else fmt for c in df.columns)
    (TABLES_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n", encoding="utf-8")


def save_json(obj: Any, name: str) -> None:
    """Dump raw numbers to results/raw/<name>.json (NumPy types converted)."""

    def convert(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    (RAW_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


class Timer:
    """Context manager that logs elapsed wall time."""

    def __init__(self, log: logging.Logger, label: str) -> None:
        self.log, self.label = log, label

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.log.info("%s finished in %.2f s", self.label, time.perf_counter() - self.t0)


__all__ = ["FIGURES_DIR", "RAW_DIR", "ROOT", "TABLES_DIR", "Setup", "Timer", "save_json", "save_table", "setup"]
