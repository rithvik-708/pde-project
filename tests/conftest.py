"""Shared fixtures."""

import numpy as np
import pytest

from backward_heat import Grid, PaperExample, SeparableExponentialSource


@pytest.fixture(scope="session")
def grid() -> Grid:
    return Grid(128)


@pytest.fixture(scope="session")
def small_grid() -> Grid:
    return Grid(48)


@pytest.fixture(scope="session")
def example() -> PaperExample:
    return PaperExample()


@pytest.fixture(scope="session")
def exact_source(example, grid) -> SeparableExponentialSource:
    return SeparableExponentialSource(example.f_spatial_coefficients(grid), example.time_rate())


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
