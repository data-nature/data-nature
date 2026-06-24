from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datetime as _dt
import os

import ee  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

from data_nature.viz.charts import (  # noqa: E402
    SEV_BG, SEV_COLOR, STATUS_COLOR, anomaly_timeseries_chart,
)
from data_nature.stats.anomaly import (  # noqa: E402
    BASELINE_START, BASELINE_END, _classify_severity, DEFAULT_THRESHOLDS,
)

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Anomaly Detection")

page_hero(
    title="🚨 Anomaly Detection & Alerts",
    subtitle="Land Surface Temperature anomalies detected via z-score analysis against a historical per-site baseline. Filter by site, date, or severity.",
    pills=["🌡️ LST z-Score", f"📅 2000–{_dt.date.today().year}", "🔴 3 Severity Levels", "📍 8 Sites"],
    emoji="🚨",
)

# ── Data loading ──────────────────────────────────────────────────────────────

_PROCESSED = _ROOT / "data" / "processed"
_MOCK = _ROOT / "data" / "mock"


def _build_from_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (ts_df, anom_df) from real processed CSVs.

    ts_df  — date, site, lst, baseline_mean, baseline_std
    anom_df — date, site, lst, baseline, z_score, severity, status, ndvi_change
    """
    from data_nature.stats import compute_zscores, detect_anomalies

    monthly = pd.read_csv(_PROCESSED / "site_monthly.csv")
    baselines = pd.read_csv(_PROCESSED / "site_baselines.csv")

    # Build date column (first of month)
    monthly["date"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    # Timeseries: merge monthly LST with leave-one-out baselines
    ts_df = (
        monthly[["date", "site", "year", "month", "lst"]]
        .merge(
            baselines[["site", "year", "month", "baseline_mean", "baseline_std"]],
            on=["site", "year", "month"],
            how="left",
        )
        .drop(columns=["year", "month"])
        .sort_values(["site", "date"])
        .reset_index(drop=True)
    )

    # Anomaly detection using the stats module
    enriched = compute_zscores(monthly)
    # detect_anomalies needs date, site, lst, ndvi, month columns
    anom_df = detect_anomalies(
        enriched[["date", "site", "lst", "ndvi", "month", "year"]],
    )
    anom_df["date"] = pd.to_datetime(anom_df["date"])

    return ts_df, anom_df


@st.cache_data(show_spinner="Loading anomaly data…")
def _load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Return (ts_df, anom_df, using_real_data)."""
    try:
        ts_df, anom_df = _build_from_processed()
        return ts_df, anom_df, True
    except Exception:
        pass

    ts_df = pd.read_csv(_MOCK / "lst_timeseries.csv", parse_dates=["date"])
    anom_df = pd.read_csv(_MOCK / "anomalies.csv", parse_dates=["date"])
    return ts_df, anom_df, False


# ── GEE initialisation ────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _init_gee() -> bool:
    try:
        key_json = st.secrets.get("gee", {}).get("key_json") or os.environ.get("GEE_KEY_JSON", "")
        sa_email = st.secrets.get("gee", {}).get("service_account") or os.environ.get("GEE_SERVICE_ACCOUNT", "")
        if key_json and sa_email:
            ee.Initialize(ee.ServiceAccountCredentials(sa_email, key_data=key_json), project="datanature")
            return True
    except Exception:
        pass
    try:
        ee.Initialize(project="datanature")
        return True
    except Exception:
        return False


GEE_OK = _init_gee()


@st.cache_data(ttl=3600, show_spinner="Fetching live MODIS data from GEE…")
def _fetch_live_months(missing_months: tuple[tuple[int, int], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (year, month) not yet in the CSV, sample LST + NDVI from GEE
    at all 8 sites, compute z-scores against the 2000-2015 baseline, and
    return new ts rows and any anomalies found.
    """
    locs = pd.read_csv(_PROCESSED / "site_locations.csv")
    monthly_hist = pd.read_csv(_PROCESSED / "site_monthly.csv")

    # Precompute 2000-2015 baselines per site/month
    ref = monthly_hist[(monthly_hist["year"] >= BASELINE_START) & (monthly_hist["year"] <= BASELINE_END)]
    lst_base = ref.groupby(["site", "month"])["lst"].agg(baseline="mean", baseline_std="std").reset_index()
    ndvi_base = ref.groupby(["site", "month"])["ndvi"].agg(ndvi_mean="mean", ndvi_std="std").reset_index()

    new_ts_rows, new_anom_rows = [], []

    for y, mo in missing_months:
        end_mo = mo % 12 + 1
        end_y = y + (1 if mo == 12 else 0)
        start = f"{y}-{mo:02d}-01"
        end = f"{end_y}-{end_mo:02d}-01"

        lst_img = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterDate(start, end)
            .select("LST_Day_1km")
            .mean()
            .multiply(0.02)
            .subtract(273.15)
        )

        # NDVI: fall back to most recent if current month unavailable
        ndvi_col = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).select("NDVI")
        if ndvi_col.size().getInfo() == 0:
            ndvi_col = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI").sort("system:time_start", False).limit(1)
        ndvi_img = ndvi_col.mean().multiply(0.0001)

        # Get previous month's NDVI per site for ndvi_change
        prev = pd.Timestamp(year=y, month=mo, day=1) - pd.DateOffset(months=1)
        prev_ndvi = monthly_hist[
            (monthly_hist["year"] == prev.year) & (monthly_hist["month"] == prev.month)
        ][["site", "ndvi"]].set_index("site")["ndvi"].to_dict()

        for _, loc in locs.iterrows():
            site = loc["site"]
            pt = ee.Geometry.Point([float(loc["lng"]), float(loc["lat"])])
            try:
                lst_val = lst_img.sample(pt, scale=1000).first().get("LST_Day_1km").getInfo()
                ndvi_val = ndvi_img.sample(pt, scale=250).first().get("NDVI").getInfo()
            except Exception:
                continue
            if lst_val is None:
                continue

            lst_val = round(float(lst_val), 2)
            ndvi_val = round(float(ndvi_val), 4) if ndvi_val is not None else None
            date = pd.Timestamp(year=y, month=mo, day=1)

            # Baseline for this site/month
            b = lst_base[(lst_base["site"] == site) & (lst_base["month"] == mo)]
            baseline = float(b["baseline"].iloc[0]) if not b.empty else None
            baseline_std = float(b["baseline_std"].iloc[0]) if not b.empty else None

            new_ts_rows.append({
                "date": date, "site": site, "lst": lst_val,
                "baseline_mean": baseline, "baseline_std": baseline_std,
            })

            if baseline is not None and baseline_std and baseline_std > 0:
                z = round((lst_val - baseline) / baseline_std, 4)
                sev = _classify_severity(z, DEFAULT_THRESHOLDS)
                if sev:
                    ndvi_change = round(ndvi_val - prev_ndvi[site], 4) if ndvi_val and site in prev_ndvi else None
                    new_anom_rows.append({
                        "date": date, "site": site, "lst": lst_val,
                        "baseline": baseline, "z_score": z,
                        "severity": sev, "status": "New",
                        "ndvi_change": ndvi_change,
                    })

    ts_new = pd.DataFrame(new_ts_rows) if new_ts_rows else pd.DataFrame(columns=["date", "site", "lst", "baseline_mean", "baseline_std"])
    anom_new = pd.DataFrame(new_anom_rows) if new_anom_rows else pd.DataFrame(columns=["date", "site", "lst", "baseline", "z_score", "severity", "status", "ndvi_change"])
    return ts_new, anom_new


ts_df, anom_df, _REAL_DATA = _load()

# ── Append live GEE data for months not yet in the CSV ───────────────────────

if GEE_OK and _REAL_DATA:
    _today = _dt.date.today()
    _latest = pd.Timestamp(ts_df["date"].max())
    _missing = []
    _cursor = _latest + pd.DateOffset(months=1)
    while _cursor.date() < _today.replace(day=1):
        _missing.append((_cursor.year, _cursor.month))
        _cursor += pd.DateOffset(months=1)

    if _missing:
        _live_ts, _live_anom = _fetch_live_months(tuple(_missing))
        if not _live_ts.empty:
            ts_df = pd.concat([ts_df, _live_ts], ignore_index=True).sort_values(["site", "date"]).reset_index(drop=True)
        if not _live_anom.empty:
            anom_df = pd.concat([anom_df, _live_anom], ignore_index=True).sort_values("date").reset_index(drop=True)
            st.info(f"🛰️ {len(_live_anom)} live anomaly event(s) detected from GEE for months not yet in the dataset.")

    # ── Live current-month metrics ────────────────────────────────────────────
    _now = _dt.date.today()
    _cur_month_ts, _ = _fetch_live_months(((_now.year, _now.month),))

    if not _cur_month_ts.empty:
        _month_name = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][_now.month]
        section_label(f"🛰️ Live — {_month_name} {_now.year} (current month)")

        _sorted_rows = _cur_month_ts.sort_values("site").reset_index(drop=True)
        _cards_html = ""
        for _row_start in (0, 4):
            _cards_html += '<div class="kpi-row">'
            for _, _row in _sorted_rows.iloc[_row_start:_row_start + 4].iterrows():
                _z = None
                if pd.notna(_row.get("baseline_mean")) and pd.notna(_row.get("baseline_std")) and _row["baseline_std"] > 0:
                    _z = (_row["lst"] - _row["baseline_mean"]) / _row["baseline_std"]

                if _z is None:
                    _color = "#1C1B18"
                    _badge = ""
                elif _z >= 3.5:
                    _color = "#DC2626"
                    _badge = " 🔴"
                elif _z >= 2.5:
                    _color = "#EA580C"
                    _badge = " 🟠"
                elif _z >= 1.5:
                    _color = "#CA8A04"
                    _badge = " 🟡"
                else:
                    _color = "#166534"
                    _badge = " ✅"

                _z_str = f"{_z:+.2f}σ" if _z is not None else "—"
                _site_label = _row["site"].replace("_", " ")
                _cards_html += f"""
                <div class="kpi-card">
                  <div class="kpi-label">{_site_label}</div>
                  <div class="kpi-value" style="font-size:1.3em;color:{_color}">{_row['lst']:.1f}°C{_badge}</div>
                  <div class="kpi-sub">z = {_z_str}</div>
                </div>"""
            _cards_html += "</div>"
        st.markdown(_cards_html, unsafe_allow_html=True)
        st.caption(f"Live LST from MODIS MOD11A1 sampled at each site for {_month_name} {_now.year}. z-score vs 2010–2023 baseline. ✅ Normal · 🟡 Warning · 🟠 Severe · 🔴 Critical")

SITES = ["All sites"] + sorted(ts_df["site"].unique())

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .kpi-row { display: flex; gap: 14px; margin-bottom: 4px; flex-wrap: wrap; }
    .kpi-card {
      flex: 1; min-width: 110px;
      background: #fff; border: 1px solid #e5e7eb;
      border-radius: 12px; padding: 18px 16px; text-align: center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .kpi-label { font-size: 0.65em; font-weight: 700; letter-spacing: 0.1em;
                 text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }
    .kpi-value { font-size: 2em; font-weight: 800; color: #1C1B18; line-height: 1; }
    .kpi-sub   { font-size: 0.72em; color: #6b7280; margin-top: 4px; }

    .anom-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
    .anom-scroll {
      max-height: 420px; overflow-y: auto;
      border: 1px solid #e5e7eb; border-radius: 10px;
    }
    .anom-table { border-radius: 0; }
    .anom-table thead th { position: sticky; top: 0; z-index: 1; }
    .anom-table th {
      background: #F0FDF4; color: #166534;
      font-weight: 700; font-size: 0.70em; letter-spacing: 0.07em;
      text-transform: uppercase; padding: 9px 12px;
      border-bottom: 2px solid #BBF7D0; text-align: left;
    }
    .anom-table td { padding: 9px 12px; border-bottom: 1px solid #F3F4F6;
                     color: #374151; vertical-align: middle; }
    .anom-table tr:hover td { background: #F9FAFB; }
    .sev-badge {
      display: inline-block; border-radius: 999px;
      padding: 2px 10px; font-size: 0.75em; font-weight: 700;
    }
    .status-dot {
      display: inline-block; width: 8px; height: 8px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }

    .explain-panel {
      background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
      border: 1px solid #A5D6A7; border-radius: 14px;
      padding: 24px 26px;
    }
    .explain-header {
      font-size: 0.68em; font-weight: 700; letter-spacing: 0.1em;
      text-transform: uppercase; color: #2E7D32; margin-bottom: 16px;
    }
    .explain-stat-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    .explain-stat {
      flex: 1; min-width: 80px; background: #fff;
      border: 1px solid #C8E6C9; border-radius: 10px;
      padding: 12px 10px; text-align: center;
    }
    .es-label { font-size: 0.62em; color: #6b7280; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.06em; }
    .es-value { font-size: 1.25em; font-weight: 800; color: #1B5E20;
                margin: 4px 0; line-height: 1; }
    .explain-body { font-size: 0.87em; color: #374151; line-height: 1.75; }
    .explain-body p { margin: 0 0 0.7em; }
    .explain-body strong { color: #2E7D32; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Status banner ─────────────────────────────────────────────────────────────

if not _REAL_DATA:
    st.info("ℹ️ Showing mock data — place real CSVs in `data/processed/` to use live data.")

# ── Filters ───────────────────────────────────────────────────────────────────

section_label("Filters")

# Severity labels present in data — include both "Warning" (real) and "Mild" (mock)
_sev_opts = ["All", "Critical", "Severe", "Warning", "Mild"]
_sev_present = [s for s in _sev_opts if s == "All" or s in anom_df["severity"].unique()]

col_site, col_sev, col_status, col_date = st.columns([2, 1.5, 1.5, 2])
with col_site:
    site_filter = st.selectbox("Site", SITES)
with col_sev:
    sev_filter = st.selectbox("Severity", _sev_present)
with col_status:
    status_filter = st.selectbox("Status", ["All", "New", "Reviewed", "Handled"])
with col_date:
    min_d = anom_df["date"].min().date()
    max_d = anom_df["date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Apply filters
filtered = anom_df.copy()
if site_filter != "All sites":
    filtered = filtered[filtered["site"] == site_filter]
if sev_filter != "All":
    filtered = filtered[filtered["severity"] == sev_filter]
if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0 = pd.Timestamp(date_range[0])
    d1 = pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= d0) & (filtered["date"] <= d1)]

filtered = filtered.sort_values("date", ascending=False).reset_index(drop=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────

section_label("Summary")

total = len(filtered)
n_critical = (filtered["severity"] == "Critical").sum()
n_severe = (filtered["severity"] == "Severe").sum()
n_new = (filtered["status"] == "New").sum()
avg_z = filtered["z_score"].mean() if total else 0.0
max_z = filtered["z_score"].max() if total else 0.0

st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Total Anomalies</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-sub">filtered events</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Critical</div>
        <div class="kpi-value" style="color:#DC2626">{n_critical}</div>
        <div class="kpi-sub">z ≥ 3.5σ</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Severe</div>
        <div class="kpi-value" style="color:#EA580C">{n_severe}</div>
        <div class="kpi-sub">2.5σ ≤ z &lt; 3.5σ</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Needs Review</div>
        <div class="kpi-value" style="color:#2E7D32">{n_new}</div>
        <div class="kpi-sub">status = New</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg z-Score</div>
        <div class="kpi-value">{avg_z:.2f}</div>
        <div class="kpi-sub">σ above baseline</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak z-Score</div>
        <div class="kpi-value">{max_z:.2f}</div>
        <div class="kpi-sub">worst event</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main two-column layout ────────────────────────────────────────────────────

st.write("")
col_table, col_explain = st.columns([3, 2], gap="large")

# ── Left: anomaly table + drill-down selector ─────────────────────────────────

with col_table:
    section_label(f"Anomaly Events ({total})")

    if total == 0:
        st.info("No anomalies match the current filters.")
        selected_idx = None
    else:
        rows_html = ""
        for _, row in filtered.head(80).iterrows():
            sev = row["severity"]
            status = row["status"]
            bg = SEV_BG.get(sev, "#fff")
            fc = SEV_COLOR.get(sev, "#374151")
            sc = STATUS_COLOR.get(status, "#6B7280")
            ndvi_val = row.get("ndvi_change", float("nan"))
            ndvi_str = f"{ndvi_val:+.3f}" if pd.notna(ndvi_val) else "—"
            rows_html += (
                f"<tr>"
                f'<td>{row["date"].strftime("%b %Y")}</td>'
                f'<td>{row["site"]}</td>'
                f'<td>{row["lst"]:.1f}°C</td>'
                f'<td>{row["baseline"]:.1f}°C</td>'
                f"<td><b>{row['z_score']:.2f}σ</b></td>"
                f'<td><span class="sev-badge" style="background:{bg};color:{fc}">{sev}</span></td>'
                f'<td><span class="status-dot" style="background:{sc}"></span>{status}</td>'
                f'<td style="color:#DC2626">{ndvi_str}</td>'
                "</tr>"
            )

        st.markdown(
            f"""
            <div class="anom-scroll">
              <table class="anom-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Site</th><th>LST</th><th>Baseline</th>
                    <th>z-Score</th><th>Severity</th><th>Status</th><th>ΔNDVI</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            {"<p style='font-size:0.75em;color:#9ca3af;margin-top:8px'>Showing top 80 events</p>" if total > 80 else ""}
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        section_label("Drill-down")
        event_labels = [
            f"{row['date'].strftime('%b %Y')} · {row['site']} · {row['severity']} ({row['z_score']:.2f}σ)"
            for _, row in filtered.iterrows()
        ]
        selected_idx = st.selectbox(
            "Select anomaly to inspect",
            range(len(event_labels)),
            format_func=lambda i: event_labels[i],
        )

# ── Right: explanation panel ──────────────────────────────────────────────────

with col_explain:
    section_label("Event Details")

    if total == 0 or selected_idx is None:
        st.markdown(
            '<div class="explain-panel">'
            '<div style="text-align:center;padding:40px 0;color:#9ca3af;font-size:0.88em;line-height:1.7">'
            "No anomaly selected.<br>Adjust filters to see events."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        sel = filtered.iloc[selected_idx]
        sev = sel["severity"]
        status = sel["status"]
        bg = SEV_BG.get(sev, "#fff")
        fc = SEV_COLOR.get(sev, "#374151")
        sc = STATUS_COLOR.get(status, "#6B7280")
        delta_t = sel["lst"] - sel["baseline"]
        ndvi_change = sel.get("ndvi_change", float("nan"))
        ndvi_str = f"{ndvi_change:+.3f}" if pd.notna(ndvi_change) else "—"

        if sev == "Critical":
            narrative = (
                f"A <strong>critical heat anomaly</strong> was recorded at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. The land surface temperature reached "
                f"<strong>{sel['lst']:.1f}°C</strong>, exceeding the historical baseline by "
                f"<strong>+{delta_t:.1f}°C</strong> ({sel['z_score']:.2f}σ)."
                "<p>At this severity, prolonged heat stress may trigger vegetation die-off and soil "
                f"moisture depletion. NDVI change: <strong>{ndvi_str}</strong>. "
                "Immediate field inspection is recommended.</p>"
            )
        elif sev == "Severe":
            narrative = (
                f"A <strong>severe temperature spike</strong> was detected at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. LST hit <strong>{sel['lst']:.1f}°C</strong>, "
                f"<strong>+{delta_t:.1f}°C</strong> above the historical mean ({sel['z_score']:.2f}σ)."
                "<p>Severe anomalies at this site have historically coincided with reduced canopy "
                f"reflectance. NDVI change: <strong>{ndvi_str}</strong>. "
                "Schedule a follow-up satellite pass within 7 days.</p>"
            )
        else:
            narrative = (
                f"A <strong>warning-level temperature anomaly</strong> occurred at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. Observed LST: <strong>{sel['lst']:.1f}°C</strong> "
                f"(+{delta_t:.1f}°C, {sel['z_score']:.2f}σ above baseline)."
                "<p>Warning anomalies are common during dry spells and typically resolve within one month. "
                f"NDVI change: <strong>{ndvi_str}</strong>. Continue monitoring; escalate if sustained.</p>"
            )

        st.markdown(
            f"""
            <div class="explain-panel">
              <div class="explain-header">⚡ Anomaly Explanation</div>
              <div class="explain-stat-row">
                <div class="explain-stat">
                  <div class="es-label">LST</div>
                  <div class="es-value">{sel["lst"]:.1f}°</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">Baseline</div>
                  <div class="es-value">{sel["baseline"]:.1f}°</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">z-Score</div>
                  <div class="es-value">{sel["z_score"]:.2f}σ</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">ΔNDVI</div>
                  <div class="es-value" style="color:#DC2626">{ndvi_str}</div>
                </div>
              </div>
              <div style="margin-bottom:14px">
                <span class="sev-badge" style="background:{bg};color:{fc}">{sev}</span>
                &nbsp;
                <span style="font-size:0.75em;color:{sc};font-weight:600">
                  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:{sc};margin-right:4px;vertical-align:middle"></span>
                  {status}
                </span>
              </div>
              <div class="explain-body"><p>{narrative}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── LST time-series chart ─────────────────────────────────────────────────────

st.write("")

if total > 0 and selected_idx is not None:
    chart_site = filtered.iloc[selected_idx]["site"]
elif site_filter != "All sites":
    chart_site = site_filter
else:
    chart_site = sorted(ts_df["site"].unique())[0]

section_label(f"LST Time-Series with Anomalies — {chart_site}")

site_ts = ts_df[ts_df["site"] == chart_site].sort_values("date").copy()
site_ts["date"] = pd.to_datetime(site_ts["date"])
site_anom = anom_df[anom_df["site"] == chart_site].copy()
site_anom["date"] = pd.to_datetime(site_anom["date"])

sel_event = None
if total > 0 and selected_idx is not None:
    ev = filtered.iloc[selected_idx]
    if ev["site"] == chart_site:
        sel_event = ev

fig = anomaly_timeseries_chart(site_ts, site_anom, selected_event=sel_event)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Monthly LST record for **{chart_site}** (2000–{_dt.date.today().year}). "
    "Coloured markers = detected anomalies (red=Critical, orange=Severe, yellow=Warning). "
    "★ = currently selected event. Shaded bands show ±1σ / ±2σ historical range."
)

