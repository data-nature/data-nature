"""
ecology.py — Ecological and heat-budget models.

Models
------
1. LotkaVolterra (DN-B5)
   Predator-prey competition between native and invasive vegetation.
   Governing equations (RK4 integration):
       dN/dt = r_n * N * (1 - (N + α*I) / K_n)
       dI/dt = r_i * I * (1 - (I + β*N) / K_i)

2. LogisticGrowth (DN-A6 / DN-B5)
   Single-population vegetation dynamics with carrying capacity.
   Governing equation (RK4 integration):
       dP/dt = r * P * (1 - P/K)

3. EnergyFlow (DN-A6)
   Surface heat-budget model driven by solar forcing, modified by NDVI.
   Governing equation (Euler integration):
       dT/dt = [Q_in(t) - k_cool * (1 + β_ndvi * NDVI) * (T - T_amb)] / C
   where:
       Q_in(t) = Q_solar * (1 - albedo) * (1 + amp * sin(2π * t / period))
   Equilibrium temperature:
       T_eq = T_amb + Q_solar*(1-albedo) / (k_cool*(1 + β_ndvi*NDVI))
   Higher NDVI → stronger cooling → lower T_eq (consistent with DN-A2 regression).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Lotka-Volterra competition model  (DN-B5)
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
            k1N, k1I = self._derivatives(N, I)
            k2N, k2I = self._derivatives(N + dt/2 * k1N, I + dt/2 * k1I)
            k3N, k3I = self._derivatives(N + dt/2 * k2N, I + dt/2 * k2I)
            k4N, k4I = self._derivatives(N + dt    * k3N, I + dt    * k3I)

            N = max(0.0, N + dt/6 * (k1N + 2*k2N + 2*k3N + k4N))
            I = max(0.0, I + dt/6 * (k1I + 2*k2I + 2*k3I + k4I))

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
# 2. Logistic growth model  (DN-A6 / DN-B5)
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


# ---------------------------------------------------------------------------
# 3. Energy-flow / heat-budget model  (DN-A6)
# ---------------------------------------------------------------------------

@dataclass
class EnergyFlow:
    """Surface heat-budget model driven by solar forcing, modulated by NDVI.

    Governing equation (Euler integration, daily time steps):
        dT/dt = [Q_in(t) - k_cool * (1 + β_ndvi * NDVI) * (T - T_amb)] / C

    Solar forcing (seasonal sinusoid):
        Q_in(t) = Q_solar * (1 - albedo) * (1 + amp * sin(2π * t / period))

    Instantaneous equilibrium temperature:
        T_eq(t) = T_amb + Q_in(t) / (k_cool * (1 + β_ndvi * NDVI))

    Parameters
    ----------
    T0 : float
        Initial surface temperature (°C).
    Q_solar : float
        Mean incoming solar irradiance (W m⁻²).
    albedo : float
        Surface shortwave albedo (0–1).
    k_cool : float
        Combined radiative + convective cooling coefficient (W m⁻² °C⁻¹).
    beta_ndvi : float
        NDVI amplifier for evapotranspirative cooling.
        Higher NDVI → stronger cooling → lower equilibrium T.
    ndvi : float
        Vegetation index of the modelled surface (0–1).
    T_amb : float
        Ambient (air) temperature (°C).
    C : float
        Effective heat capacity of the surface layer (J m⁻² °C⁻¹ / scaling).
    amp : float
        Fractional amplitude of seasonal solar variation.
    period : float
        Seasonal period in days (default 365).
    """

    T0:        float = 35.0
    Q_solar:   float = 450.0
    albedo:    float = 0.20
    k_cool:    float = 8.0
    beta_ndvi: float = 3.0
    ndvi:      float = 0.4
    T_amb:     float = 25.0
    C:         float = 50.0
    amp:       float = 0.25
    period:    float = 365.0

    def _q_in(self, t: float) -> float:
        return self.Q_solar * (1 - self.albedo) * (
            1 + self.amp * np.sin(2 * np.pi * t / self.period)
        )

    def _t_eq(self, q_in: float) -> float:
        return self.T_amb + q_in / (self.k_cool * (1 + self.beta_ndvi * self.ndvi))

    def simulate(self, steps: int = 365, dt: float = 1.0) -> pd.DataFrame:
        """Integrate the heat-budget ODE using forward Euler.

        Parameters
        ----------
        steps : int
            Number of time steps (days by default).
        dt : float
            Time step size (days).

        Returns
        -------
        pd.DataFrame with columns: t, T, T_eq, Q_in
        """
        t_vals   = np.zeros(steps + 1)
        T_vals   = np.zeros(steps + 1)
        Teq_vals = np.zeros(steps + 1)
        Qin_vals = np.zeros(steps + 1)

        T = self.T0
        cooling = self.k_cool * (1 + self.beta_ndvi * self.ndvi)

        for k in range(steps + 1):
            t = k * dt
            q = self._q_in(t)
            t_vals[k]   = t
            T_vals[k]   = T
            Teq_vals[k] = self._t_eq(q)
            Qin_vals[k] = q
            if k < steps:
                dT = (q - cooling * (T - self.T_amb)) / self.C
                T += dt * dT

        return pd.DataFrame({
            "t":    t_vals,
            "T":    T_vals,
            "T_eq": Teq_vals,
            "Q_in": Qin_vals,
        })


def simulate_energy_flow(
    model: EnergyFlow | None = None,
    steps: int = 365,
    dt: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """Run an EnergyFlow simulation and return a trajectory DataFrame.

    Parameters
    ----------
    model : EnergyFlow, optional
        Pre-configured model.  If None, built from ``**kwargs``.
    steps, dt
        Passed to model.simulate().

    Returns
    -------
    pd.DataFrame with columns: t, T, T_eq, Q_in
    """
    if model is None:
        model = EnergyFlow(**kwargs)
    return model.simulate(steps=steps, dt=dt)
