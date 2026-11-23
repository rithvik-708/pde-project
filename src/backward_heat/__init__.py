"""Reproduction of H. T. Nguyen, *Stability and regularization for 2-D backward heat
conduction equation*, Math. Reports 17(67), 1 (2015), 1-23.

Fourier-truncation regularisation of the 2-D backward heat problem on (0, pi)^2.
"""

from .analytical import PaperExample
from .config import DEFAULT_CONFIG, ExperimentConfig
from .fourier import Grid, sine_coefficients, synthesize
from .noise import add_paper_noise, make_rng
from .reconstruction import ReconstructionResult, reconstruct
from .regularization import cutoff_rules, regularized_coefficients
from .source import GridSource, SeparableExponentialSource, ZeroSource

__all__ = [
    "DEFAULT_CONFIG",
    "ExperimentConfig",
    "Grid",
    "GridSource",
    "PaperExample",
    "ReconstructionResult",
    "SeparableExponentialSource",
    "ZeroSource",
    "add_paper_noise",
    "cutoff_rules",
    "make_rng",
    "reconstruct",
    "regularized_coefficients",
    "sine_coefficients",
    "synthesize",
]
__version__ = "1.0.0"
