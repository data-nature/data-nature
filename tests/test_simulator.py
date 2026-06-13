"""
tests/test_simulator.py — DN-B5 tests for cellular automaton + ecology models
==============================================================================
All tests run on synthetic data — no processed CSVs required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_nature.models.cellular_automaton import (
    COOLING,
    GRID_SIZE,
    SAFE_LST,
    HeatCA,
    init_grid,
    make_veg_mask,
    run_simulation,
)
from data_nature.models.ecology import (
    LogisticGrowth,
    LotkaVolterra,
    simulate_logistic,
    simulate_lv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_monthly(sites=None) -> pd.DataFrame:
    """Minimal synthetic site_monthly DataFrame for CA grid init."""
    if sites is None:
        sites = ["Site_A", "Site_B"]
    rows = []
    for site in sites:
        for month in range(1, 13):
            t = (month - 1) / 12 * 2 * np.pi
            rows.append({
                "site":  site,
                "month": month,
                "lst":   25 + 10 * np.sin(t),
                "ndvi":  0.5 - 0.2 * np.sin(t),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def monthly_df():
    return _synthetic_monthly()


@pytest.fixture
def ca_default(monthly_df):
    """A default HeatCA with +20% vegetation, 0 steps taken."""
    lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=42)
    ca = HeatCA(lst0, ndvi0, season="Summer")
    ca.apply_vegetation(20.0)
    return ca


# ---------------------------------------------------------------------------
# HeatCA — grid initialisation
# ---------------------------------------------------------------------------

class TestInitGrid:
    def test_output_shape(self, monthly_df):
        lst, ndvi = init_grid("Site_A", "Summer", monthly_df)
        assert lst.shape  == (GRID_SIZE, GRID_SIZE)
        assert ndvi.shape == (GRID_SIZE, GRID_SIZE)

    def test_ndvi_clamped(self, monthly_df):
        _, ndvi = init_grid("Site_A", "Winter", monthly_df, seed=0)
        assert ndvi.min() >= 0.01
        assert ndvi.max() <= 0.99

    def test_lst_finite(self, monthly_df):
        lst, _ = init_grid("Site_A", "Summer", monthly_df)
        assert np.isfinite(lst).all()

    def test_reproducible_with_seed(self, monthly_df):
        lst1, _ = init_grid("Site_A", "Summer", monthly_df, seed=7)
        lst2, _ = init_grid("Site_A", "Summer", monthly_df, seed=7)
        np.testing.assert_array_equal(lst1, lst2)

    def test_different_seeds_differ(self, monthly_df):
        lst1, _ = init_grid("Site_A", "Summer", monthly_df, seed=1)
        lst2, _ = init_grid("Site_A", "Summer", monthly_df, seed=2)
        assert not np.array_equal(lst1, lst2)

    def test_unknown_site_raises(self, monthly_df):
        with pytest.raises(ValueError, match="No data"):
            init_grid("NoSuchSite", "Summer", monthly_df)


# ---------------------------------------------------------------------------
# HeatCA — vegetation mask
# ---------------------------------------------------------------------------

class TestVegMask:
    def test_mask_shape(self):
        mask = make_veg_mask(GRID_SIZE)
        assert mask.shape == (GRID_SIZE, GRID_SIZE)

    def test_mask_is_boolean(self):
        mask = make_veg_mask(GRID_SIZE)
        assert mask.dtype == bool

    def test_mask_centre_is_true(self):
        mask = make_veg_mask(GRID_SIZE)
        cy = cx = GRID_SIZE // 2
        assert mask[cy, cx], "Centre cell must be in the planting zone"

    def test_mask_corners_are_false(self):
        mask = make_veg_mask(GRID_SIZE)
        assert not mask[0, 0]
        assert not mask[0, -1]
        assert not mask[-1, 0]
        assert not mask[-1, -1]


# ---------------------------------------------------------------------------
# HeatCA — apply_vegetation
# ---------------------------------------------------------------------------

class TestApplyVegetation:
    def test_positive_veg_increases_ndvi(self, monthly_df):
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=0)
        ca = HeatCA(lst0, ndvi0)
        ndvi_before = ca.ndvi.copy()
        ca.apply_vegetation(20.0)
        mask = ca._veg_mask
        # NDVI inside mask must have increased
        assert (ca.ndvi[mask] >= ndvi_before[mask]).all()

    def test_negative_veg_decreases_ndvi(self, monthly_df):
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=0)
        ca = HeatCA(lst0, ndvi0)
        ndvi_before = ca.ndvi.copy()
        ca.apply_vegetation(-30.0)
        mask = ca._veg_mask
        assert (ca.ndvi[mask] <= ndvi_before[mask]).all()

    def test_zero_veg_no_change(self, monthly_df):
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=0)
        ca = HeatCA(lst0, ndvi0)
        ndvi_before = ca.ndvi.copy()
        ca.apply_vegetation(0.0)
        np.testing.assert_array_almost_equal(ca.ndvi, ndvi_before)

    def test_ndvi_stays_clamped(self, monthly_df):
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=0)
        ca = HeatCA(lst0, ndvi0)
        ca.apply_vegetation(10000.0)   # extreme vegetation
        assert ca.ndvi.max() <= 0.99
        assert ca.ndvi.min() >= 0.01


# ---------------------------------------------------------------------------
# HeatCA — step() physics
# ---------------------------------------------------------------------------

class TestHeatCAStep:
    def test_step_reduces_lst_with_positive_veg(self, ca_default):
        """Adding vegetation should cool the grid on average."""
        lst_before = ca_default.lst.mean()
        ca_default.step()
        lst_after = ca_default.lst.mean()
        assert lst_after < lst_before, (
            f"Expected cooling, got warming: {lst_after:.3f} > {lst_before:.3f}"
        )

    def test_step_appends_to_history(self, ca_default):
        n_before = len(ca_default.history)
        ca_default.step()
        assert len(ca_default.history) == n_before + 1

    def test_step_appends_to_trajectory(self, ca_default):
        n_before = len(ca_default.avg_lst_trajectory)
        ca_default.step()
        assert len(ca_default.avg_lst_trajectory) == n_before + 1

    def test_lst_finite_after_steps(self, ca_default):
        for _ in range(10):
            ca_default.step()
        assert np.isfinite(ca_default.lst).all()

    def test_higher_ndvi_cells_cooler_neighbours(self, monthly_df):
        """Key physics assertion: higher-NDVI cells cool their neighbours."""
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=42)
        ca = HeatCA(lst0, ndvi0, season="Summer")
        ca.apply_vegetation(50.0)
        for _ in range(10):
            ca.step()
        # The centre (high NDVI after planting) should be cooler than initial
        cy = cx = GRID_SIZE // 2
        assert ca.lst[cy, cx] < lst0[cy, cx], (
            "Centre cell (high NDVI) should be cooler than its initial value"
        )

    def test_cooling_propagates_to_neighbours(self, monthly_df):
        """Neighbours of planted cells must cool measurably over steps."""
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=42)
        ca = HeatCA(lst0, ndvi0, season="Summer")
        ca.apply_vegetation(50.0)
        for _ in range(15):
            ca.step()
        delta = ca.lst - lst0
        # At least 25% of cells should show cooling
        cooled_fraction = (delta < 0).mean()
        assert cooled_fraction >= 0.25, (
            f"Only {cooled_fraction:.0%} of cells cooled — expected ≥25%"
        )


# ---------------------------------------------------------------------------
# HeatCA — metrics
# ---------------------------------------------------------------------------

class TestHeatCAMetrics:
    def test_delta_lst_negative_with_positive_veg(self, ca_default):
        for _ in range(5):
            ca_default.step()
        assert ca_default.delta_lst() < 0

    def test_new_safe_cells_non_negative(self, ca_default):
        for _ in range(5):
            ca_default.step()
        assert ca_default.new_safe_cells() >= 0

    def test_history_length_matches_steps(self, monthly_df):
        lst0, ndvi0 = init_grid("Site_A", "Summer", monthly_df, seed=0)
        ca = HeatCA(lst0, ndvi0)
        ca.apply_vegetation(20)
        n = 7
        for _ in range(n):
            ca.step()
        assert len(ca.history) == n + 1   # step 0 + n steps


# ---------------------------------------------------------------------------
# run_simulation convenience wrapper
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_returns_heat_ca(self, monthly_df):
        ca = run_simulation("Site_A", "Summer", monthly_df, n_steps=3, seed=0)
        assert isinstance(ca, HeatCA)

    def test_correct_number_of_steps(self, monthly_df):
        n = 6
        ca = run_simulation("Site_A", "Winter", monthly_df, n_steps=n, seed=0)
        assert len(ca.history) == n + 1


# ---------------------------------------------------------------------------
# LotkaVolterra
# ---------------------------------------------------------------------------

class TestLotkaVolterra:
    def test_output_columns(self):
        df = simulate_lv(steps=50, dt=0.1)
        assert set(df.columns) == {"t", "native", "invasive"}

    def test_output_length(self):
        steps = 100
        df = simulate_lv(steps=steps, dt=0.1)
        assert len(df) == steps + 1

    def test_populations_non_negative(self):
        df = simulate_lv(steps=200, dt=0.1)
        assert (df["native"]   >= 0).all()
        assert (df["invasive"] >= 0).all()

    def test_populations_finite(self):
        df = simulate_lv(steps=200, dt=0.1)
        assert np.isfinite(df["native"].values).all()
        assert np.isfinite(df["invasive"].values).all()

    def test_competitive_exclusion(self):
        """With high invasive growth rate and strong competition, invasive wins."""
        lv = LotkaVolterra(
            N0=0.8, I0=0.2,
            r_native=0.2, r_invasive=0.8,
            K_native=1.0, K_invasive=1.0,
            alpha=1.5, beta=0.3,
        )
        df = simulate_lv(lv, steps=500, dt=0.1)
        assert df["invasive"].iloc[-1] > df["native"].iloc[-1], (
            "Invasive species should outcompete native under these parameters"
        )

    def test_coexistence(self):
        """With weak competition, both populations coexist at steady state."""
        lv = LotkaVolterra(
            N0=0.5, I0=0.5,
            r_native=0.4, r_invasive=0.4,
            K_native=1.0, K_invasive=1.0,
            alpha=0.5, beta=0.5,
        )
        df = simulate_lv(lv, steps=500, dt=0.1)
        # Both populations should be alive at the end
        assert df["native"].iloc[-1]   > 0.1
        assert df["invasive"].iloc[-1] > 0.1

    def test_time_is_monotonic(self):
        df = simulate_lv(steps=100, dt=0.1)
        assert (df["t"].diff().dropna() > 0).all()

    def test_initial_conditions_respected(self):
        lv = LotkaVolterra(N0=0.6, I0=0.3)
        df = simulate_lv(lv, steps=10)
        assert df["native"].iloc[0]   == pytest.approx(0.6)
        assert df["invasive"].iloc[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# LogisticGrowth
# ---------------------------------------------------------------------------

class TestLogisticGrowth:
    def test_output_columns(self):
        df = simulate_logistic(steps=100)
        assert set(df.columns) == {"t", "population"}

    def test_output_length(self):
        steps = 80
        df = simulate_logistic(steps=steps)
        assert len(df) == steps + 1

    def test_saturates_at_carrying_capacity(self):
        """Population must approach K (within 5%) at steady state."""
        lg = LogisticGrowth(P0=0.05, r=0.5, K=1.0)
        df = simulate_logistic(lg, steps=300, dt=0.1)
        assert df["population"].iloc[-1] >= 0.95 * lg.K, (
            f"Expected population ≥ 0.95 * K={lg.K}, "
            f"got {df['population'].iloc[-1]:.4f}"
        )

    def test_population_non_negative(self):
        df = simulate_logistic(steps=200)
        assert (df["population"] >= 0).all()

    def test_population_monotone_from_below_K(self):
        """Starting below K, population must increase monotonically."""
        lg = LogisticGrowth(P0=0.1, r=0.4, K=1.0)
        df = simulate_logistic(lg, steps=200, dt=0.1)
        diffs = df["population"].diff().dropna()
        assert (diffs >= -1e-9).all(), "Population decreased unexpectedly"

    def test_initial_condition_respected(self):
        lg = LogisticGrowth(P0=0.3, r=0.5, K=1.0)
        df = simulate_logistic(lg, steps=10)
        assert df["population"].iloc[0] == pytest.approx(0.3)

    def test_does_not_exceed_carrying_capacity(self):
        lg = LogisticGrowth(P0=0.1, r=0.5, K=0.8)
        df = simulate_logistic(lg, steps=500, dt=0.1)
        assert df["population"].max() <= lg.K + 1e-6