"""Tests for DN-A6 ecological models: LogisticGrowth and EnergyFlow."""

from __future__ import annotations

import numpy as np
import pytest

from data_nature.models.ecology import (
    EnergyFlow,
    LogisticGrowth,
    simulate_energy_flow,
    simulate_logistic,
)
from data_nature.viz.charts import energy_flow_chart, logistic_growth_chart


# ── LogisticGrowth ────────────────────────────────────────────────────────────


class TestLogisticGrowth:
    def test_output_columns(self):
        df = LogisticGrowth().simulate()
        assert set(df.columns) == {"t", "population"}

    def test_output_length(self):
        steps = 80
        df = LogisticGrowth().simulate(steps=steps)
        assert len(df) == steps + 1

    def test_starts_at_P0(self):
        P0 = 0.05
        df = LogisticGrowth(P0=P0).simulate()
        assert df["population"].iloc[0] == pytest.approx(P0)

    def test_saturates_at_carrying_capacity(self):
        """Key acceptance criterion: curve must reach ≥ 95% of K."""
        K = 1.0
        df = LogisticGrowth(P0=0.05, r=0.5, K=K).simulate(steps=300, dt=0.1)
        assert df["population"].iloc[-1] >= 0.95 * K, (
            f"Final population {df['population'].iloc[-1]:.4f} < 0.95*K={0.95*K}"
        )

    def test_saturates_nonstandard_K(self):
        K = 2.5
        df = LogisticGrowth(P0=0.1, r=0.3, K=K).simulate(steps=400, dt=0.1)
        assert df["population"].iloc[-1] >= 0.95 * K

    def test_population_never_exceeds_K(self):
        K = 1.0
        df = LogisticGrowth(P0=0.1, r=0.8, K=K).simulate(steps=200, dt=0.1)
        assert df["population"].max() <= K + 1e-6

    def test_population_monotonically_increasing(self):
        """With P0 < K the curve must be non-decreasing."""
        df = LogisticGrowth(P0=0.1, r=0.4, K=1.0).simulate(steps=200, dt=0.1)
        diffs = df["population"].diff().dropna()
        assert (diffs >= -1e-9).all()

    def test_time_starts_at_zero(self):
        df = LogisticGrowth().simulate()
        assert df["t"].iloc[0] == pytest.approx(0.0)

    def test_simulate_logistic_convenience(self):
        df = simulate_logistic(P0=0.2, r=0.5, K=1.0, steps=50)
        assert "population" in df.columns
        assert len(df) == 51

    def test_simulate_logistic_passes_model(self):
        m = LogisticGrowth(P0=0.3, r=0.6, K=1.0)
        df = simulate_logistic(model=m, steps=50)
        assert df["population"].iloc[0] == pytest.approx(0.3)

    def test_higher_r_grows_faster(self):
        """Higher growth rate should reach 80% of K sooner."""
        def steps_to_80pct(r: float) -> int:
            df = LogisticGrowth(P0=0.05, r=r, K=1.0).simulate(steps=500, dt=0.1)
            mask = df["population"] >= 0.80
            return int(mask.idxmax()) if mask.any() else 9999

        assert steps_to_80pct(0.8) < steps_to_80pct(0.3)


# ── EnergyFlow ────────────────────────────────────────────────────────────────


class TestEnergyFlow:
    def test_output_columns(self):
        df = EnergyFlow().simulate()
        assert set(df.columns) == {"t", "T", "T_eq", "Q_in"}

    def test_output_length(self):
        steps = 100
        df = EnergyFlow().simulate(steps=steps)
        assert len(df) == steps + 1

    def test_starts_at_T0(self):
        T0 = 40.0
        df = EnergyFlow(T0=T0).simulate()
        assert df["T"].iloc[0] == pytest.approx(T0)

    def test_converges_toward_equilibrium(self):
        """Surface temp should move toward equilibrium, not diverge."""
        m = EnergyFlow(T0=50.0, Q_solar=400.0, k_cool=8.0, ndvi=0.3,
                       T_amb=25.0, C=50.0, amp=0.0)
        df = m.simulate(steps=500)
        # With amp=0, T_eq is constant; T should converge toward it
        T_eq_const = df["T_eq"].iloc[0]
        final_err = abs(df["T"].iloc[-1] - T_eq_const)
        initial_err = abs(df["T"].iloc[0] - T_eq_const)
        assert final_err < initial_err, "Temperature did not converge toward equilibrium"

    def test_higher_ndvi_lower_equilibrium(self):
        """More vegetation → stronger cooling → lower mean T_eq."""
        low_ndvi  = EnergyFlow(ndvi=0.1, amp=0.0).simulate(steps=365)
        high_ndvi = EnergyFlow(ndvi=0.7, amp=0.0).simulate(steps=365)
        assert high_ndvi["T_eq"].mean() < low_ndvi["T_eq"].mean()

    def test_higher_ndvi_lower_steady_state_T(self):
        """After convergence, higher NDVI site should be cooler."""
        low  = EnergyFlow(ndvi=0.1, T0=35.0, amp=0.0).simulate(steps=1000)
        high = EnergyFlow(ndvi=0.8, T0=35.0, amp=0.0).simulate(steps=1000)
        assert high["T"].iloc[-1] < low["T"].iloc[-1]

    def test_q_in_seasonal_variation(self):
        """With amp > 0, Q_in should vary sinusoidally (min < mean < max)."""
        df = EnergyFlow(amp=0.3, period=365.0).simulate(steps=365)
        assert df["Q_in"].max() > df["Q_in"].mean() > df["Q_in"].min()

    def test_q_in_constant_when_amp_zero(self):
        df = EnergyFlow(amp=0.0).simulate(steps=100)
        assert df["Q_in"].std() == pytest.approx(0.0, abs=1e-9)

    def test_t_eq_always_above_ambient(self):
        """Equilibrium T must always exceed ambient (solar heating is positive)."""
        df = EnergyFlow(T_amb=20.0, Q_solar=400.0, albedo=0.2).simulate(steps=365)
        assert (df["T_eq"] > 20.0).all()

    def test_simulate_energy_flow_convenience(self):
        df = simulate_energy_flow(T0=30.0, ndvi=0.4, steps=50)
        assert "T" in df.columns
        assert len(df) == 51

    def test_simulate_energy_flow_passes_model(self):
        m = EnergyFlow(T0=45.0)
        df = simulate_energy_flow(model=m, steps=10)
        assert df["T"].iloc[0] == pytest.approx(45.0)


# ── Chart helpers ─────────────────────────────────────────────────────────────


class TestEcologyCharts:
    def test_logistic_growth_chart_returns_figure(self):
        import plotly.graph_objects as go
        df = simulate_logistic(P0=0.1, r=0.4, K=1.0, steps=100)
        fig = logistic_growth_chart(df, K=1.0)
        assert isinstance(fig, go.Figure)

    def test_logistic_growth_chart_has_traces(self):
        df = simulate_logistic(P0=0.1, r=0.4, K=1.0, steps=100)
        fig = logistic_growth_chart(df, K=1.0)
        assert len(fig.data) >= 1

    def test_energy_flow_chart_returns_figure(self):
        import plotly.graph_objects as go
        df = simulate_energy_flow(steps=100)
        fig = energy_flow_chart(df)
        assert isinstance(fig, go.Figure)

    def test_energy_flow_chart_with_planted_overlay(self):
        df1 = simulate_energy_flow(ndvi=0.3, steps=100)
        df2 = simulate_energy_flow(ndvi=0.6, steps=100)
        fig = energy_flow_chart(df1, df_planted=df2)
        # Should have 3 traces: equilibrium, current T, planted T
        assert len(fig.data) == 3

    def test_energy_flow_chart_without_overlay(self):
        df = simulate_energy_flow(steps=100)
        fig = energy_flow_chart(df)
        # Should have 2 traces: equilibrium + current T
        assert len(fig.data) == 2
