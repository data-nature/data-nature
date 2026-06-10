"""
cellular_automaton.py — DN-B5 Heat-diffusion cellular automaton
================================================================
Models how vegetation changes propagate heat reduction across a spatial grid
of land patches.  Each cell holds an LST and NDVI value; transition rules are
driven by local NDVI, cell temperature, and Moore-neighbourhood diffusion.

Public API
----------
    HeatCA          — main CA class (init → step → trajectory)
    init_grid       — build a site/season grid from real site_monthly data
    run_simulation  — convenience wrapper: run n steps and return all grids

Physics
-------
    1. Vegetation change shifts NDVI in the planting zone.
    2. Each cell's LST relaxes toward its NDVI-adjusted thermal equilibrium.
    3. Moore-neighbourhood (8-cell) diffusion spreads heat to adjacent cells.
    4. A small seasonal offset is added each step.

Key parameters (all tunable at construction time)
--------------------------------------------------
    COOLING = 9.0   °C cooling per +1.0 NDVI unit
    DIFF    = 0.30  fraction of cell heat diffused to neighbours per step
    RELAX   = 0.35  relaxation rate toward vegetation-adjusted equilibrium
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRID_SIZE  = 10
SAFE_LST   = 30.0   # °C — threshold for a "safe" (cool) cell
COOLING    = 9.0    # °C cooling per +1.0 NDVI
DIFF       = 0.30   # heat diffusion fraction per step
RELAX      = 0.35   # relaxation toward equilibrium per step

SEASON_MONTHS: dict[str, list[int]] = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}
SEASON_HEAT: dict[str, float] = {
    "Spring":  0.00,
    "Summer":  0.40,
    "Autumn": -0.10,
    "Winter": -0.30,
}


# ---------------------------------------------------------------------------
# Grid initialisation from real data
# ---------------------------------------------------------------------------

def init_grid(
    site: str,
    season: str,
    monthly_df: pd.DataFrame,
    grid_size: int = GRID_SIZE,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reproducible grid from real site_monthly statistics.

    The centre of the grid is cooler / greener (higher NDVI) and the edges
    are warmer / drier — a realistic urban-heat-island / forest-edge pattern.

    Parameters
    ----------
    site : str
        Site name matching monthly_df["site"].
    season : str
        One of "Spring", "Summer", "Autumn", "Winter".
    monthly_df : pd.DataFrame
        Real (or mock) site_monthly DataFrame with columns: site, month, lst, ndvi.
    grid_size : int
        Side length of the square grid (default 10).
    seed : int, optional
        RNG seed for reproducibility.  Defaults to a hash of site+season.

    Returns
    -------
    (lst_grid, ndvi_grid) — both shape (grid_size, grid_size), float64.
    """
    months = SEASON_MONTHS[season]
    rows   = monthly_df[
        (monthly_df["site"] == site) & (monthly_df["month"].isin(months))
    ]
    if rows.empty:
        raise ValueError(
            f"No data for site='{site}', season='{season}'. "
            f"Available sites: {monthly_df['site'].unique().tolist()}"
        )

    base_lst  = float(rows["lst"].mean())
    base_ndvi = float(rows["ndvi"].mean())

    if seed is None:
        seed = sum(ord(c) for c in site + season) % (2 ** 31)
    rng = np.random.default_rng(seed)

    cy = cx = (grid_size - 1) / 2.0
    lst_g  = np.empty((grid_size, grid_size))
    ndvi_g = np.empty((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) / (grid_size * 0.65)
            lst_g[i, j]  = base_lst  + 3.5 * d + rng.normal(0.0, 0.6)
            ndvi_g[i, j] = base_ndvi - 0.10 * d + rng.normal(0.0, 0.022)

    return lst_g, np.clip(ndvi_g, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Vegetation planting mask
# ---------------------------------------------------------------------------

def make_veg_mask(grid_size: int = GRID_SIZE, radius: float = 2.5) -> np.ndarray:
    """Circular vegetation-change zone centred on the grid.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid.
    radius : float
        Radius (in cells) of the planting zone.

    Returns
    -------
    Boolean mask of shape (grid_size, grid_size).
    """
    cy = cx = (grid_size - 1) / 2.0
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - cy) ** 2 + (j - cx) ** 2 <= radius ** 2:
                mask[i, j] = True
    return mask


# ---------------------------------------------------------------------------
# Core CA class
# ---------------------------------------------------------------------------

@dataclass
class HeatCA:
    """Cellular-automaton heat-diffusion model.

    Parameters
    ----------
    lst0 : np.ndarray
        Initial LST grid, shape (G, G).
    ndvi0 : np.ndarray
        Initial NDVI grid, shape (G, G).
    season : str
        Season label controlling the per-step heat offset.
    cooling : float
        °C cooling per +1.0 NDVI unit.
    diff : float
        Heat diffusion fraction per step (Moore neighbourhood).
    relax : float
        Relaxation rate toward vegetation-adjusted equilibrium.
    safe_lst : float
        Threshold below which a cell is considered "safe" (cool).

    Examples
    --------
    >>> lst0, ndvi0 = init_grid("Carmel_Forest", "Summer", monthly_df)
    >>> ca = HeatCA(lst0, ndvi0, season="Summer")
    >>> ca.apply_vegetation(veg_pct=20)
    >>> for _ in range(5):
    ...     ca.step()
    >>> print(ca.delta_lst())   # mean LST change vs initial
    """

    lst0:    np.ndarray
    ndvi0:   np.ndarray
    season:  str = "Summer"
    cooling: float = COOLING
    diff:    float = DIFF
    relax:   float = RELAX
    safe_lst: float = SAFE_LST

    # internal state — not part of the constructor signature
    _lst:     np.ndarray = field(init=False, repr=False)
    _ndvi:    np.ndarray = field(init=False, repr=False)
    _lst_eq:  np.ndarray = field(init=False, repr=False)
    _history: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _avgs:    list[float]      = field(default_factory=list, init=False, repr=False)
    _veg_mask: np.ndarray      = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g = self.lst0.shape[0]
        self._lst      = self.lst0.copy()
        self._ndvi     = self.ndvi0.copy()
        self._lst_eq   = self.lst0.copy()   # overwritten by apply_vegetation
        self._veg_mask = make_veg_mask(g)
        self._history  = [self._lst.copy()]
        self._avgs     = [float(self._lst.mean())]

    # ── Public interface ───────────────────────────────────────────────────

    def apply_vegetation(self, veg_pct: float) -> "HeatCA":
        """Shift NDVI in the central planting zone by veg_pct%.

        Recomputes the thermal equilibrium grid used by step().
        Call this once before running step() calls.

        Parameters
        ----------
        veg_pct : float
            Percentage change in vegetation cover.
            Positive = plant more; negative = remove vegetation.

        Returns
        -------
        self  (fluent interface)
        """
        ndvi_new = self._ndvi.copy()
        ndvi_new[self._veg_mask] = np.clip(
            ndvi_new[self._veg_mask] * (1.0 + veg_pct / 100.0), 0.01, 0.99
        )
        self._ndvi  = ndvi_new
        self._lst_eq = self.lst0 - self.cooling * (ndvi_new - self.ndvi0)
        return self

    def step(self) -> "HeatCA":
        """Advance the automaton by one generation.

        Physics applied in order:
          1. Relax LST toward vegetation-adjusted equilibrium.
          2. Moore-neighbourhood heat diffusion.
          3. Add seasonal heat offset.

        Returns
        -------
        self  (fluent interface)
        """
        # 1. relaxation
        self._lst = self._lst + self.relax * (self._lst_eq - self._lst)
        # 2. diffusion
        self._lst = self._diffuse(self._lst)
        # 3. seasonal offset
        self._lst = self._lst + SEASON_HEAT[self.season] * 0.02

        self._history.append(self._lst.copy())
        self._avgs.append(float(self._lst.mean()))
        return self

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def lst(self) -> np.ndarray:
        """Current LST grid."""
        return self._lst

    @property
    def ndvi(self) -> np.ndarray:
        """Current NDVI grid."""
        return self._ndvi

    @property
    def history(self) -> list[np.ndarray]:
        """All LST grids from step 0 (initial) to current step."""
        return self._history

    @property
    def avg_lst_trajectory(self) -> list[float]:
        """Mean LST at each step, including step 0."""
        return self._avgs

    def delta_lst(self) -> float:
        """Mean LST change (current − initial), °C."""
        return float(self._lst.mean() - self.lst0.mean())

    def max_delta(self) -> float:
        """Maximum cell-level LST change (can be positive or negative)."""
        return float((self._lst - self.lst0).max())

    def min_delta(self) -> float:
        """Minimum cell-level LST change."""
        return float((self._lst - self.lst0).min())

    def new_safe_cells(self) -> int:
        """Number of cells that crossed below safe_lst since step 0."""
        was_unsafe = self.lst0     >= self.safe_lst
        now_safe   = self._lst     < self.safe_lst
        return int((was_unsafe & now_safe).sum())

    # ── Internal helpers ───────────────────────────────────────────────────

    def _diffuse(self, lst: np.ndarray) -> np.ndarray:
        """Moore-neighbourhood heat diffusion (pure numpy, no scipy)."""
        g = lst.shape[0]
        padded = np.pad(lst, 1, mode="edge")
        s = np.zeros_like(lst)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                s += padded[1 + di: g + 1 + di, 1 + dj: g + 1 + dj]
        return (1 - self.diff) * lst + self.diff * (s / 8.0)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_simulation(
    site:       str,
    season:     str,
    monthly_df: pd.DataFrame,
    veg_pct:    float = 20.0,
    n_steps:    int   = 5,
    seed:       Optional[int] = None,
) -> HeatCA:
    """Run a full CA simulation and return the HeatCA object.

    Parameters
    ----------
    site, season, monthly_df
        Passed to init_grid.
    veg_pct : float
        Vegetation cover change percentage (default +20 %).
    n_steps : int
        Number of CA steps to run.
    seed : int, optional
        RNG seed for the grid initialisation.

    Returns
    -------
    HeatCA
        Fully run automaton with history and trajectory populated.
    """
    lst0, ndvi0 = init_grid(site, season, monthly_df, seed=seed)
    ca = HeatCA(lst0, ndvi0, season=season)
    ca.apply_vegetation(veg_pct)
    for _ in range(n_steps):
        ca.step()
    return ca