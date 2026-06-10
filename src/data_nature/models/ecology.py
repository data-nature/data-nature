"""
ecology.py — DN-B5/DN-A6 Ecological models
===========================================
Implements two ecological models used in the Simulator page:

1. **Lotka-Volterra** (predator–prey) — models interacting native vs invasive
   vegetation populations competing for the same niche.

2. **Logistic growth** — single-population vegetation dynamics with carrying
   capacity (used by DN-A6 in the Heatmap page).

Public API
----------
    LotkaVolterra       — native/invasive vegetation competition model
    simulate_lv         — convenience: run LV and return trajectory DataFrame
    LogisticGrowth      — single-population logistic model
    simulate_logistic   — convenience: run logistic and return trajectory DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Lotka-Volterra competition model
# ---------------------------------------------------------------------------

@dataclass
class LotkaVolterra:
    """Lotka-Volterra competition model for native vs invasive vegetation.

    Governing equations (continuous, integrated with RK4)
    -------------------------------------------------------
        dN/dt = r_n * N * (1 - (N + α * I) / K_n)
        dI/dt = r_i * I * (1 - (I + β * N) / K_i)

    where:
        N  — native vegetation density
        I  — invasive vegetation density
        r_n, r_i    — intrinsic growth rates
        K_n, K_i    — carrying capacities
        α           — competition effect of invasive on native
        β           — competition effect of native on invasive

    Parameters
    ----------
    N0 : float
        Initial native vegetation density (default 0.8).
    I0 : float
        Initial invasive vegetation density (default 0.2).
    r_native : float
        Intrinsic growth rate of native vegetation (default 0.3).
    r_invasive : float
        Intrinsic growth rate of invasive vegetation (default 0.5).
    K_native : float
        Carrying capacity for native vegetation (default 1.0).
    K_invasive : float
        Carrying capacity for invasive vegetation (default 1.0).
    alpha : float
        Competition coefficient: effect of invasive on native (default 1.2).
    beta : float
        Competition coefficient: effect of native on invasive (default 0.8).

    Examples
    --------
    >>> lv = LotkaVolterra(N0=0.8, I0=0.2)
    >>> df = simulate_lv(lv, steps=200, dt=0.1)
    >>> df[["t", "native", "invasive"]].tail()
    """

    N0:          float = 0.8
    I0:          float = 0.2
    r_native:    float = 0.30
    r_invasive:  float = 0.50
    K_native:    float = 1.0
    K_invasive:  float = 1.0
    alpha:       float = 1.2   # invasive → native competition
    beta:        float = 0.8   # native   → invasive competition

    def _derivatives(self, N: float, I: float) -> tuple[float, float]:
        """Compute dN/dt and dI/dt."""
        dN = self.r_native   * N * (1 - (N + self.alpha * I) / self.K_native)
        dI = self.r_invasive * I * (1 - (I + self.beta  * N) / self.K_invasive)
        return dN, dI

    def simulate(
        self,
        steps: int = 200,
        dt:    float = 0.1,
    ) -> pd.DataFrame:
        """Integrate the LV system using 4th-order Runge-Kutta.

        Parameters
        ----------
        steps : int
            Number of time steps.
        dt : float
            Time step size.

        Returns
        -------
        pd.DataFrame with columns: t, native, invasive
        """
        t_vals = np.zeros(steps + 1)
        N_vals = np.zeros(steps + 1)
        I_vals = np.zeros(steps + 1)

        N, I = self.N0, self.I0
        N_vals[0], I_vals[0], t_vals[0] = N, I, 0.0

        for k in range(steps):
            # RK4
            k1N, k1I = self._derivatives(N, I)
            k2N, k2I = self._derivatives(N + dt/2 * k1N, I + dt/2 * k1I)
            k3N, k3I = self._derivatives(N + dt/2 * k2N, I + dt/2 * k2I)
            k4N, k4I = self._derivatives(N + dt    * k3N, I + dt    * k3I)

            N = N + dt/6 * (k1N + 2*k2N + 2*k3N + k4N)
            I = I + dt/6 * (k1I + 2*k2I + 2*k3I + k4I)

            # clamp to [0, max(K)] — populations cannot go negative
            N = max(0.0, N)
            I = max(0.0, I)

            t_vals[k+1] = (k+1) * dt
            N_vals[k+1] = N
            I_vals[k+1] = I

        return pd.DataFrame({"t": t_vals, "native": N_vals, "invasive": I_vals})


def simulate_lv(
    model:  LotkaVolterra | None = None,
    steps:  int   = 200,
    dt:     float = 0.1,
    **kwargs,
) -> pd.DataFrame:
    """Run a Lotka-Volterra simulation and return a trajectory DataFrame.

    Parameters
    ----------
    model : LotkaVolterra, optional
        Pre-configured model.  If None, a default model is created using
        any extra keyword arguments.
    steps, dt
        Passed to model.simulate().
    **kwargs
        Forwarded to LotkaVolterra() if model is None.

    Returns
    -------
    pd.DataFrame with columns: t, native, invasive
    """
    if model is None:
        model = LotkaVolterra(**kwargs)
    return model.simulate(steps=steps, dt=dt)


# ---------------------------------------------------------------------------
# 2. Logistic growth model  (used by DN-A6)
# ---------------------------------------------------------------------------

@dataclass
class LogisticGrowth:
    """Single-population logistic growth model.

    Governing equation
    ------------------
        dP/dt = r * P * (1 - P / K)

    Parameters
    ----------
    P0 : float
        Initial population density (default 0.1).
    r : float
        Intrinsic growth rate (default 0.4).
    K : float
        Carrying capacity (default 1.0).

    Examples
    --------
    >>> lg = LogisticGrowth(P0=0.05, r=0.5, K=1.0)
    >>> df = simulate_logistic(lg, steps=100, dt=0.1)
    >>> assert df["population"].iloc[-1] > 0.95 * lg.K
    """

    P0: float = 0.1
    r:  float = 0.4
    K:  float = 1.0

    def simulate(
        self,
        steps: int   = 100,
        dt:    float = 0.1,
    ) -> pd.DataFrame:
        """Integrate logistic growth using RK4.

        Returns
        -------
        pd.DataFrame with columns: t, population
        """
        t_vals = np.zeros(steps + 1)
        P_vals = np.zeros(steps + 1)

        P = self.P0
        P_vals[0] = P

        for k in range(steps):
            def _dP(p: float) -> float:
                return self.r * p * (1 - p / self.K)

            k1 = _dP(P)
            k2 = _dP(P + dt/2 * k1)
            k3 = _dP(P + dt/2 * k2)
            k4 = _dP(P + dt    * k3)

            P = max(0.0, P + dt/6 * (k1 + 2*k2 + 2*k3 + k4))
            t_vals[k+1] = (k+1) * dt
            P_vals[k+1] = P

        return pd.DataFrame({"t": t_vals, "population": P_vals})


def simulate_logistic(
    model: LogisticGrowth | None = None,
    steps: int   = 100,
    dt:    float = 0.1,
    **kwargs,
) -> pd.DataFrame:
    """Run a logistic-growth simulation and return a trajectory DataFrame.

    Parameters
    ----------
    model : LogisticGrowth, optional
        Pre-configured model.  If None, a default model is created.
    **kwargs
        Forwarded to LogisticGrowth() if model is None.

    Returns
    -------
    pd.DataFrame with columns: t, population
    """
    if model is None:
        model = LogisticGrowth(**kwargs)
    return model.simulate(steps=steps, dt=dt)