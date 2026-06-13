"""
Folium map helpers for the Heatmap page.

All functions are pure Python (no Streamlit dependency) so they are
testable and reusable by other pages.
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import folium
import pandas as pd

# ── Layer configuration ────────────────────────────────────────────────────────

LAYER_CFG: dict[str, dict] = {
    "LST Day (MOD11A1)": {
        "col": "lst",
        "vmin": 20.0,
        "vmax": 55.0,
        "unit": "°C",
        "cmap": "RdYlBu_r",
        "label": "Land Surface Temperature — Day (°C)",
        "palette": [
            "#313695", "#74add1", "#e0f3f8",
            "#fee090", "#f46d43", "#d73027", "#a50026",
        ],
        "yearly": False,
    },
    "NDVI (MOD13Q1)": {
        "col": "ndvi",
        "vmin": 0.0,
        "vmax": 0.8,
        "unit": "",
        "cmap": "RdYlGn",
        "label": "Vegetation Index — NDVI",
        "palette": [
            "#d73027", "#f46d43", "#fdae61", "#fee08b",
            "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850",
        ],
        "yearly": False,
    },
    "Land Cover (MOD12Q1)": {
        "col": "ndvi",
        "vmin": 0.0,
        "vmax": 17.0,
        "unit": "",
        "cmap": "tab20",
        "label": "Land Cover Type 1 (IGBP)",
        "palette": [
            "1c0dff", "05450a", "086a10", "54a708", "78d203",
            "009900", "c6b044", "dcd159", "dade48", "fbff13",
            "b6ff05", "27af87", "c24f44", "a5a5a5", "ff6d4c",
            "69fff8", "f9ffa4", "1c0dff",
        ],
        "yearly": True,
    },
}


# ── Colour helpers ─────────────────────────────────────────────────────────────


def hex_color(val: float, vmin: float, vmax: float, cmap_name: str) -> str:
    """Map a scalar value to a hex colour string using a matplotlib colormap."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    return mcolors.to_hex(cmap(norm(float(np.clip(val, vmin, vmax)))))


def legend_html(cfg: dict) -> str:
    """Return an HTML gradient legend bar for the given layer config."""
    palette = cfg["palette"]
    clean = [c if c.startswith("#") else f"#{c}" for c in palette]
    gradient = ", ".join(clean)
    vmin, vmax, unit = cfg["vmin"], cfg["vmax"], cfg["unit"]
    mid = (vmin + vmax) / 2
    return (
        f'<div style="margin:8px 0 18px">'
        f'<div style="font-size:0.65em;font-weight:700;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#6b7280;margin-bottom:5px">{cfg["label"]}</div>'
        f'<div style="height:14px;border-radius:7px;'
        f'background:linear-gradient(90deg,{gradient});border:1px solid #e5e7eb"></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:3px;'
        f'font-size:0.68em;color:#6b7280">'
        f"<span>{vmin:.0f}{unit}</span><span>{mid:.0f}{unit}</span>"
        f"<span>{vmax:.0f}{unit}</span>"
        f"</div></div>"
    )


# ── Map builder ────────────────────────────────────────────────────────────────


def build_site_map(
    snap_df: pd.DataFrame,
    layer_cfg: dict,
    highlight: str,
    tile_url: str | None = None,
) -> folium.Map:
    """
    Build a Folium map centred on Northern Israel with site markers.

    Parameters
    ----------
    snap_df : pd.DataFrame
        Rows for one time slice, merged with site locations.
        Must have columns: site, lat, lng, lst, ndvi, delta, is_anomaly.
    layer_cfg : dict
        One entry from LAYER_CFG — supplies col, vmin, vmax, cmap.
    highlight : str
        Site name to highlight, or "All" for no highlight.
    tile_url : str | None
        GEE tile URL to overlay.  If None, the map shows only the base tile.
    """
    fmap = folium.Map(
        location=[32.78, 35.30],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if tile_url:
        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine / MODIS / NASA",
            name=layer_cfg["label"],
            opacity=0.88,
            overlay=True,
        ).add_to(fmap)

    for _, row in snap_df.iterrows():
        val = float(row[layer_cfg["col"]])
        fill = hex_color(val, layer_cfg["vmin"], layer_cfg["vmax"], layer_cfg["cmap"])
        is_hi = highlight != "All" and row["site"] == highlight
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=11 if is_hi else 8,
            color="#FFD700" if is_hi else "#ffffff",
            weight=3 if is_hi else 2,
            fill=True,
            fill_color=fill,
            fill_opacity=0.92,
            tooltip=folium.Tooltip(
                f"<b>{row['site']}</b><br>"
                f"LST: {row['lst']:.1f}°C &nbsp;·&nbsp; NDVI: {row['ndvi']:.3f}<br>"
                f"Δ z: {row['delta']:+.2f}σ &nbsp;·&nbsp; "
                f"{'⚠️ Anomaly' if row['is_anomaly'] else '✅ Normal'}",
                sticky=True,
            ),
            popup=folium.Popup(str(row["site"]), max_width=200),
        ).add_to(fmap)

    return fmap
