"""Centralised configuration for the reproduction of Nguyen (2015).

Every numerical value used by the experiments lives here, so the experiment
scripts never hard-code parameters. Values marked [paper] are taken from
Section 4 of the paper; values marked [inferred] were recovered from the
paper's own tables (see ``docs/paper_analysis.md``); values marked [choice]
are implementation decisions that the paper does not specify.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
RAW_DIR: Path = RESULTS_DIR / "raw"
LOGS_DIR: Path = RESULTS_DIR / "logs"


def ensure_result_dirs() -> None:
    """Create the ``results/`` sub-directories if they do not exist."""
    for directory in (FIGURES_DIR, TABLES_DIR, RAW_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- configs
@dataclass(frozen=True)
class GridConfig:
    """Uniform grid x_i = i*pi/I, y_j = j*pi/J, i, j = 0..I  [paper, p.18]."""

    n_intervals: int = 128  # I = J = 128 [paper]
    length: float = float(np.pi)  # Omega = (0, pi)^2 [paper]

    def __post_init__(self) -> None:
        if self.n_intervals < 2:
            raise ValueError("n_intervals must be >= 2")
        if self.length <= 0:
            raise ValueError("length must be positive")


@dataclass(frozen=True)
class ProblemConfig:
    """Test problem (24) of the paper."""

    final_time: float = 1.0  # T = 1 [paper]
    n_terms: int = 15  # sum k = 1..15 [paper]


@dataclass(frozen=True)
class RegularizationConfig:
    """Parameters of the cut-off rules M^i_eps."""

    beta: float = 0.5  # Theorem 7, beta = 1/2 [paper]
    r: float = 2.0  # Theorem 6, r > 0 -- NOT stated; r = 2 [inferred from Table 1]
    m4_sqrt: bool = True  # Theorem 7 statement uses sqrt -- matches Table 1 [inferred]


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to rerun the published experiments."""

    grid: GridConfig = field(default_factory=GridConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    noise_levels: tuple[float, ...] = (1e-3, 1e-5, 1e-7)  # [paper]
    times: tuple[float, ...] = tuple(j / 10 for j in range(10))  # t = j/10, j = 0..9 [paper]
    seed: int = 2015  # [choice] -- the paper gives no seed
    figure_noise_level: float = 1e-5  # Figures 1-2 use eps = 1e-5 [paper]
    section_cut_indices: tuple[int, ...] = (51, 64, 102)  # y = 51pi/128, pi/2, 51pi/64 [paper]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (saved next to every result)."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the configuration as pretty JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


DEFAULT_CONFIG = ExperimentConfig()

# Published values, used for side-by-side comparison [paper, Tables 1-2].
PAPER_TABLE1: dict[str, dict[float, float]] = {
    "M1": {1e-3: 4.840, 1e-5: 8.986, 1e-7: 13.28},
    "M2": {1e-3: 3.454, 1e-5: 5.756, 1e-7: 8.059},
    "M3": {1e-3: 8.705, 1e-5: 13.87, 1e-7: 18.84},
    "M4": {1e-3: 2.146, 1e-5: 2.770, 1e-7: 3.278},
}

# Table 2, rows t = 0.0..0.9; columns (eps, rule). Transcribed verbatim, including
# the typo 1.287E-02 at (eps=1e-5, M3, t=0.1) -- see paper_analysis.md, A6.
_T2_COLUMNS = [(eps, rule) for eps in (1e-3, 1e-5, 1e-7) for rule in ("M1", "M2", "M3", "M4")]
_T2_ROWS = [
    [5.161e-01, 5.161e-01, 1.423e-01, 5.161e-01, 1.423e-01, 1.423e-01, 1.424e-01, 5.161e-01, 1.423e-01, 1.423e-01, 1.424e-01, 5.161e-01],
    [4.670e-01, 4.670e-01, 1.288e-01, 4.670e-01, 1.287e-01, 1.287e-01, 1.287e-02, 4.670e-01, 1.287e-01, 1.287e-01, 1.288e-01, 4.670e-01],
    [4.226e-01, 4.226e-01, 1.165e-01, 4.226e-01, 1.165e-01, 1.165e-01, 1.165e-01, 4.226e-01, 1.165e-01, 1.165e-01, 1.165e-01, 4.226e-01],
    [3.824e-01, 3.824e-01, 1.054e-01, 3.824e-01, 1.054e-01, 1.054e-01, 1.054e-01, 3.824e-01, 1.054e-01, 1.054e-01, 1.054e-01, 3.824e-01],
    [3.460e-01, 3.460e-01, 9.537e-02, 3.460e-01, 9.537e-02, 9.537e-02, 9.537e-02, 3.460e-01, 9.537e-02, 9.537e-02, 9.537e-02, 3.460e-01],
    [3.130e-01, 3.130e-01, 8.629e-02, 3.130e-01, 8.629e-02, 8.629e-02, 8.629e-02, 3.130e-01, 8.629e-02, 8.629e-02, 8.629e-02, 3.130e-01],
    [2.833e-01, 2.833e-01, 7.808e-02, 2.833e-01, 7.808e-02, 7.808e-02, 7.808e-02, 2.833e-01, 7.808e-02, 7.808e-02, 7.808e-02, 2.833e-01],
    [2.563e-01, 2.563e-01, 7.065e-02, 2.563e-01, 7.065e-02, 7.065e-02, 7.065e-02, 2.563e-01, 7.065e-02, 7.065e-02, 7.065e-02, 2.563e-01],
    [2.319e-01, 2.319e-01, 6.393e-02, 2.319e-01, 6.393e-02, 6.393e-02, 6.393e-02, 2.319e-01, 6.393e-02, 6.393e-02, 6.393e-02, 2.319e-01],
    [2.098e-01, 2.098e-01, 5.784e-02, 2.098e-01, 5.784e-02, 5.784e-02, 5.784e-02, 2.098e-01, 5.784e-02, 5.784e-02, 5.784e-02, 2.098e-01],
]
PAPER_TABLE2: dict[tuple[float, str], dict[float, float]] = {
    col: {round(j / 10, 1): _T2_ROWS[j][c] for j in range(10)} for c, col in enumerate(_T2_COLUMNS)
}
