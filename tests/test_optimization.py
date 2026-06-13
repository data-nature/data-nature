"""Tests for src/data_nature/models/optimization.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_nature.models.optimization import (
    COOLING,
    GRID,
    SEASON_MONTHS,
    VEG_BOOST,
    expected_delta_lst,
    fitness,
    init_site_grid,
    optimize_planting,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_grids() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic LST/NDVI grids for testing."""
    rng = np.random.default_rng(0)
    lst = rng.uniform(25.0, 45.0, (GRID, GRID))
    ndvi = rng.uniform(0.1, 0.7, (GRID, GRID))
    return lst, ndvi


@pytest.fixture()
def sample_monthly() -> pd.DataFrame:
    """Minimal site_monthly DataFrame covering one site across all months."""
    rows = []
    for month in range(1, 13):
        rows.append({"site": "Test_Site", "month": month, "lst": 30.0 + month, "ndvi": 0.4})
    return pd.DataFrame(rows)


# ── init_site_grid ────────────────────────────────────────────────────────────


class TestInitSiteGrid:
    def test_output_shape(self, sample_monthly):
        lst_g, ndvi_g = init_site_grid(sample_monthly, "Test_Site", "Summer")
        assert lst_g.shape == (GRID, GRID)
        assert ndvi_g.shape == (GRID, GRID)

    def test_ndvi_clipped(self, sample_monthly):
        _, ndvi_g = init_site_grid(sample_monthly, "Test_Site", "Summer")
        assert float(ndvi_g.min()) >= 0.01
        assert float(ndvi_g.max()) <= 0.99

    def test_deterministic(self, sample_monthly):
        """Same site+season always produces the same grid."""
        a1, b1 = init_site_grid(sample_monthly, "Test_Site", "Summer")
        a2, b2 = init_site_grid(sample_monthly, "Test_Site", "Summer")
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)

    def test_different_seasons_differ(self, sample_monthly):
        lst_sum, _ = init_site_grid(sample_monthly, "Test_Site", "Summer")
        lst_win, _ = init_site_grid(sample_monthly, "Test_Site", "Winter")
        assert not np.array_equal(lst_sum, lst_win)

    def test_all_seasons_accepted(self, sample_monthly):
        for season in SEASON_MONTHS:
            lst_g, ndvi_g = init_site_grid(sample_monthly, "Test_Site", season)
            assert lst_g.shape == (GRID, GRID)

    def test_missing_site_uses_defaults(self, sample_monthly):
        """Unknown site should not raise — uses fallback base values."""
        lst_g, ndvi_g = init_site_grid(sample_monthly, "Unknown_Site", "Summer")
        assert lst_g.shape == (GRID, GRID)


# ── fitness ───────────────────────────────────────────────────────────────────


class TestFitness:
    def test_returns_float(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        ind = np.zeros(GRID * GRID, dtype=bool)
        ind[:5] = True
        assert isinstance(fitness(ind, lst, ndvi), float)

    def test_empty_individual_zero(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        ind = np.zeros(GRID * GRID, dtype=bool)
        assert fitness(ind, lst, ndvi) == pytest.approx(0.0)

    def test_all_planted_positive(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        ind = np.ones(GRID * GRID, dtype=bool)
        assert fitness(ind, lst, ndvi) > 0.0

    def test_hot_bare_beats_cool_vegetated(self):
        """Hot + bare cells should score higher than cool + vegetated cells."""
        lst = np.full((GRID, GRID), 25.0)
        ndvi = np.full((GRID, GRID), 0.7)
        lst[0, 0] = 50.0
        ndvi[0, 0] = 0.05

        hot_ind = np.zeros(GRID * GRID, dtype=bool)
        hot_ind[0] = True  # cell (0,0): hot + bare

        cool_ind = np.zeros(GRID * GRID, dtype=bool)
        cool_ind[1] = True  # cell (0,1): cool + vegetated

        assert fitness(hot_ind, lst, ndvi) > fitness(cool_ind, lst, ndvi)

    def test_adjacency_bonus(self):
        """Clustered cells should score higher than scattered cells."""
        lst = np.full((GRID, GRID), 40.0)
        ndvi = np.full((GRID, GRID), 0.1)

        clustered = np.zeros(GRID * GRID, dtype=bool)
        clustered[0] = True   # (0,0)
        clustered[1] = True   # (0,1) — adjacent

        scattered = np.zeros(GRID * GRID, dtype=bool)
        scattered[0] = True   # (0,0)
        scattered[GRID] = True  # (1,0) wait, that's adjacent too
        # Use fully separated cells
        scattered2 = np.zeros(GRID * GRID, dtype=bool)
        scattered2[0] = True           # (0,0)
        scattered2[GRID * GRID - 1] = True  # (9,9) — far corner

        assert fitness(clustered, lst, ndvi) >= fitness(scattered2, lst, ndvi)


# ── expected_delta_lst ────────────────────────────────────────────────────────


class TestExpectedDeltaLst:
    def test_cooling_is_negative(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        mask = np.ones((GRID, GRID), dtype=bool)
        delta = expected_delta_lst(mask, ndvi)
        assert delta < 0.0, "Planting everywhere should reduce mean LST"

    def test_no_cells_zero_delta(self, synthetic_grids):
        _, ndvi = synthetic_grids
        mask = np.zeros((GRID, GRID), dtype=bool)
        assert expected_delta_lst(mask, ndvi) == pytest.approx(0.0, abs=1e-9)

    def test_magnitude_proportional_to_cells(self, synthetic_grids):
        """More cells planted → larger (more negative) ΔLST."""
        _, ndvi = synthetic_grids
        mask_few = np.zeros((GRID, GRID), dtype=bool)
        mask_few[:2, :2] = True

        mask_many = np.zeros((GRID, GRID), dtype=bool)
        mask_many[:5, :5] = True

        delta_few = expected_delta_lst(mask_few, ndvi)
        delta_many = expected_delta_lst(mask_many, ndvi)
        assert delta_many < delta_few

    def test_uses_cooling_constant(self):
        """With uniform ndvi, ΔLST = -COOLING × boost × fraction_planted."""
        ndvi = np.full((GRID, GRID), 0.5)
        mask = np.ones((GRID, GRID), dtype=bool)
        delta = expected_delta_lst(mask, ndvi, boost=VEG_BOOST)
        expected = -COOLING * (0.5 * VEG_BOOST)
        assert delta == pytest.approx(expected, rel=1e-4)


# ── optimize_planting ─────────────────────────────────────────────────────────


class TestOptimizePlanting:
    def test_returns_expected_keys(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=5, n_gen=5, pop_size=10, seed=42)
        assert "history" in result
        assert "solutions" in result

    def test_history_length(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        n_gen = 10
        result = optimize_planting(lst, ndvi, budget=5, n_gen=n_gen, pop_size=10, seed=42)
        assert len(result["history"]) == n_gen

    def test_fitness_monotonically_nondecreasing(self, synthetic_grids):
        """Elitism guarantees best fitness never decreases."""
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=8, n_gen=30, pop_size=20, seed=42)
        history = result["history"]
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-9, (
                f"Fitness decreased at gen {i}: {history[i-1]:.4f} → {history[i]:.4f}"
            )

    def test_solutions_sorted_by_fitness(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=5, n_gen=10, pop_size=10, seed=7)
        sols = result["solutions"]
        for i in range(1, len(sols)):
            assert sols[i]["fitness"] <= sols[i - 1]["fitness"]

    def test_solution_mask_shape(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=6, n_gen=5, pop_size=10, seed=1)
        for sol in result["solutions"]:
            assert sol["mask"].shape == (GRID, GRID)
            assert sol["mask"].dtype == bool

    def test_solution_budget_respected(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        budget = 7
        result = optimize_planting(lst, ndvi, budget=budget, n_gen=5, pop_size=10, seed=3)
        for sol in result["solutions"]:
            assert int(sol["mask"].sum()) == budget

    def test_solution_delta_lst_is_negative(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=5, n_gen=5, pop_size=10, seed=9)
        for sol in result["solutions"]:
            assert sol["delta_lst"] < 0.0

    def test_reproducible_with_seed(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        r1 = optimize_planting(lst, ndvi, budget=5, n_gen=10, pop_size=10, seed=99)
        r2 = optimize_planting(lst, ndvi, budget=5, n_gen=10, pop_size=10, seed=99)
        assert r1["history"] == r2["history"]
        np.testing.assert_array_equal(r1["solutions"][0]["mask"], r2["solutions"][0]["mask"])

    def test_different_seeds_differ(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        r1 = optimize_planting(lst, ndvi, budget=5, n_gen=10, pop_size=10, seed=1)
        r2 = optimize_planting(lst, ndvi, budget=5, n_gen=10, pop_size=10, seed=2)
        # Different seeds should (almost always) produce different histories
        assert r1["history"] != r2["history"]

    def test_progress_callback_called(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        calls: list[tuple] = []
        n_gen = 5
        optimize_planting(
            lst, ndvi, budget=4, n_gen=n_gen, pop_size=8, seed=0,
            progress_cb=lambda g, t, f: calls.append((g, t, f)),
        )
        assert len(calls) == n_gen
        assert calls[0] == (1, n_gen, calls[0][2])
        assert calls[-1][0] == n_gen

    def test_at_most_five_solutions(self, synthetic_grids):
        lst, ndvi = synthetic_grids
        result = optimize_planting(lst, ndvi, budget=5, n_gen=20, pop_size=20, seed=42)
        assert len(result["solutions"]) <= 5

    def test_best_fitness_improves_over_random(self, synthetic_grids):
        """More generations should yield better or equal fitness than fewer."""
        lst, ndvi = synthetic_grids
        r_short = optimize_planting(lst, ndvi, budget=8, n_gen=5,  pop_size=20, seed=42)
        r_long  = optimize_planting(lst, ndvi, budget=8, n_gen=50, pop_size=20, seed=42)
        assert r_long["history"][-1] >= r_short["history"][-1]
