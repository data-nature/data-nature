"""
optimization.py — Genetic algorithm for optimal vegetation planting.

Finds the GRID×GRID cells that maximise a fitness score balancing:
  - high current LST (hot spots benefit most from planting)
  - low current NDVI (bare ground has most room to improve)
  - spatial adjacency (clustered planting improves microclimate)

The expected ΔLST uses the empirical NDVI→LST coefficient from DN-A2
regression: COOLING °C per unit NDVI increase.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

GRID: int = 10
COOLING: float = 9.0      # °C cooling per unit NDVI increase (from DN-A2)
VEG_BOOST: float = 0.20   # fractional NDVI boost assumed per planted cell
_CELL_LAT: float = 0.0027
_CELL_LNG: float = 0.0032

SEASON_MONTHS: dict[str, list[int]] = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}


# ── Grid initialisation ────────────────────────────────────────────────────────


def init_site_grid(
    monthly_df: pd.DataFrame,
    site: str,
    season: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic GRID×GRID LST / NDVI grid for one site and season.

    The grid is seeded deterministically from the site+season name and
    centred on the site's seasonal mean values from the real monthly data.
    Spatial variation radiates outward from the centre (urban-heat-island
    pattern for LST, inverse for NDVI).

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Must contain ``site``, ``month``, ``lst``, ``ndvi`` columns.
    site : str
        Site name matching ``monthly_df["site"]``.
    season : str
        One of ``SEASON_MONTHS`` keys.

    Returns
    -------
    (lst_grid, ndvi_grid) : tuple of (GRID, GRID) float64 arrays
    """
    months = SEASON_MONTHS[season]
    rows = monthly_df[
        (monthly_df["site"] == site) & (monthly_df["month"].isin(months))
    ]
    base_lst = float(rows["lst"].mean()) if not rows.empty else 35.0
    base_ndvi = float(rows["ndvi"].mean()) if not rows.empty else 0.3

    seed = sum(ord(c) for c in site + season) % (2**31)
    rng = np.random.default_rng(seed)
    cy, cx = (GRID - 1) / 2.0, (GRID - 1) / 2.0
    lst_g = np.empty((GRID, GRID))
    ndvi_g = np.empty((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) / (GRID * 0.65)
            lst_g[i, j] = base_lst + 3.5 * d + rng.normal(0.0, 0.6)
            ndvi_g[i, j] = base_ndvi - 0.10 * d + rng.normal(0.0, 0.022)
    return lst_g, np.clip(ndvi_g, 0.01, 0.99)


# ── Fitness ───────────────────────────────────────────────────────────────────


def fitness(ind: np.ndarray, lst: np.ndarray, ndvi: np.ndarray) -> float:
    """
    Compute fitness for one individual (flat bool array of length GRID²).

    Fitness = Σ over planted cells of (norm_LST × (1 − NDVI))
              + adjacency bonus (contiguous clusters improve microclimate).

    Higher is better.  Best individuals are hot, bare-ground cells in
    spatially cohesive clusters.
    """
    mask = ind.reshape(GRID, GRID)
    lst_n = (lst - lst.min()) / (lst.max() - lst.min() + 1e-9)
    base = float(np.sum(lst_n[mask] * (1.0 - ndvi[mask])))
    adj = 0
    for i in range(GRID):
        for j in range(GRID):
            if mask[i, j]:
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID and 0 <= nj < GRID and mask[ni, nj]:
                        adj += 1
    return base + adj * 0.05


# ── Expected ΔLST ─────────────────────────────────────────────────────────────


def expected_delta_lst(
    mask: np.ndarray,
    ndvi: np.ndarray,
    boost: float = VEG_BOOST,
) -> float:
    """
    Predicted mean LST change (°C) if planted cells gain ``boost`` × NDVI.

    Negative value = cooling.  Uses the NDVI→LST coefficient (COOLING)
    derived from the DN-A2 linear regression.
    """
    ndvi_new = ndvi.copy()
    ndvi_new[mask] = np.clip(ndvi_new[mask] * (1.0 + boost), 0.01, 0.99)
    return -COOLING * float((ndvi_new - ndvi).mean())


# ── GA operators ──────────────────────────────────────────────────────────────


def _tournament(
    pop: list[np.ndarray],
    fits: list[float],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.choice(len(pop), k, replace=False)
    return pop[int(max(idx, key=lambda i: fits[i]))].copy()


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    combined = np.where(p1 | p2)[0]
    n = len(combined)
    chosen = (
        rng.choice(combined, budget, replace=False)
        if n >= budget
        else rng.choice(len(p1), budget, replace=False)
    )
    child = np.zeros(len(p1), dtype=bool)
    child[chosen] = True
    return child


def _mutate(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = ind.copy()
    planted = np.where(ind)[0]
    unplanted = np.where(~ind)[0]
    new[rng.choice(planted)] = False
    new[rng.choice(unplanted)] = True
    return new


# ── Public API ────────────────────────────────────────────────────────────────


def optimize_planting(
    lst_grid: np.ndarray,
    ndvi_grid: np.ndarray,
    budget: int,
    n_gen: int = 60,
    pop_size: int = 30,
    seed: int | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> dict:
    """
    Run the genetic algorithm for optimal planting location selection.

    Parameters
    ----------
    lst_grid : np.ndarray, shape (GRID, GRID)
        Current land surface temperature grid (°C).
    ndvi_grid : np.ndarray, shape (GRID, GRID)
        Current NDVI grid (0–1).
    budget : int
        Number of cells to plant (chromosome length = budget selected cells).
    n_gen : int
        Number of generations to run.
    pop_size : int
        Population size.
    seed : int | None
        RNG seed for reproducibility.  None = random.
    progress_cb : callable | None
        Optional ``(gen, n_gen, best_fitness) → None`` callback called
        after each generation (e.g. to update a UI progress bar).

    Returns
    -------
    dict with keys:

    ``"history"`` : list[float]
        Best fitness value at the end of each generation.
        Monotonically non-decreasing (elitism guarantees this).

    ``"solutions"`` : list[dict]
        Up to 5 unique best solutions, sorted by fitness descending.
        Each dict has:
        - ``"fitness"``  : float
        - ``"mask"``     : np.ndarray (GRID, GRID) bool
        - ``"delta_lst"`` : float — expected mean ΔLST in °C (negative = cooling)
    """
    G2 = GRID * GRID
    rng = np.random.default_rng(seed)

    pop: list[np.ndarray] = []
    for _ in range(pop_size):
        ind = np.zeros(G2, dtype=bool)
        ind[rng.choice(G2, budget, replace=False)] = True
        pop.append(ind)

    history: list[float] = []
    seen: set[bytes] = set()
    top5: list[dict] = []

    for gen in range(n_gen):
        fits = [fitness(ind, lst_grid, ndvi_grid) for ind in pop]
        best_fit = max(fits)
        history.append(best_fit)

        for f, ind in sorted(zip(fits, pop), key=lambda x: -x[0])[:5]:
            key = ind.tobytes()
            if key not in seen:
                seen.add(key)
                mask = ind.reshape(GRID, GRID)
                top5.append({
                    "fitness": f,
                    "mask": mask.copy(),
                    "delta_lst": expected_delta_lst(mask, ndvi_grid),
                })
                top5.sort(key=lambda x: -x["fitness"])
                top5 = top5[:5]

        # Elitism: best individual always survives
        best_idx = int(np.argmax(fits))
        new_pop: list[np.ndarray] = [pop[best_idx].copy()]

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, k=3, rng=rng)
            p2 = _tournament(pop, fits, k=3, rng=rng)
            child = _crossover(p1, p2, budget, rng)
            if rng.random() < 0.25:
                child = _mutate(child, rng)
            new_pop.append(child)

        pop = new_pop

        if progress_cb is not None:
            progress_cb(gen + 1, n_gen, best_fit)

    return {"history": history, "solutions": top5}
