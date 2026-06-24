# Pages Guide

A plain-English guide to each page in the Data Nature app — what it shows, how to use it, and what you can learn from it.

---

## 1. Heat Map

### What it is
An interactive map of Northern Israel showing satellite-measured environmental conditions across 8 ecological sites. It combines real MODIS satellite data from NASA with historical records going back to the year 2000.

### What you can see
- **Land Surface Temperature (LST)** — how hot the ground surface is, in degrees Celsius, measured from space by the MODIS MOD11A1 sensor.
- **NDVI** — a measure of vegetation greenness. Values close to 1 mean dense healthy vegetation; values close to 0 mean bare or dry land.
- **Land Cover** — the type of land use at each site (forest, urban, agricultural, wetland, arid, mixed).

### The 8 sites
Carmel Forest, Haifa Coastal Urban, Hula Valley Wetland, Jezreel Valley Agricultural, Jordan Valley Arid, Lower Galilee Mixed, Nazareth Urban, Upper Galilee Forest.

### How to use it

**Controls at the top:**
- **MODIS Layer** — switch between LST, NDVI, and Land Cover as the map overlay.
- **Year / Month sliders** — pick any month from January 2000 up to the current month. Months not yet in the database are fetched live from Google Earth Engine.
- **Highlight site** — select a specific site to highlight it on the map and load its details on the right.

**The map:**
- Colored markers show each site. The color and size reflect the selected layer's value for that month.
- Click any marker to select that site and update the detail panel on the right.
- If Google Earth Engine is authenticated, the map also shows a satellite imagery overlay as a background layer.

**Site Detail panel (right side):**
- Shows the LST, NDVI, LST z-score, NDVI z-score, and the delta between them for the selected site and month.
- A green badge means conditions are normal; a red badge means an anomaly was detected (LST z-score ≥ 1.5, meaning the temperature is significantly above the historical average for that month).

**Timeline chart (bottom):**
- Shows the full LST z-score history for the selected site from 2000 to today.
- Grey dots are normal months; red dots are anomaly months.
- The green dashed line marks the month you currently have selected.

### What you can learn from it
- Which sites are experiencing unusual heat compared to their historical norm.
- How LST and NDVI relate — a high LST z-score combined with a low NDVI z-score (high delta) can indicate vegetation stress or loss.
- How conditions at a site have changed over more than 25 years.
- Whether the current month is an outlier relative to the same month in past years.

### Live data note
Data up to May 2026 is loaded from pre-processed files. Months after that are fetched live from Google Earth Engine. For those months, LST reflects the current month's satellite readings, while NDVI uses the most recent available 16-day composite (MOD13Q1 has a slightly longer processing delay).

---

## 2. Anomaly Detection

### What it is
A detection and alerting page that identifies months where Land Surface Temperature at a site was significantly hotter than its historical norm. All data comes from real MODIS satellite observations via Google Earth Engine.

### How anomalies are detected
Each month's LST is compared against the average and standard deviation of the same calendar month across the reference period 2000–2015. The result is a z-score — how many standard deviations the observation sits above the historical norm. Using a fixed reference period (rather than a rolling average) ensures that recent warming shows up as a genuine anomaly rather than being absorbed into a shifting baseline.

**Severity levels:**
- **Warning** — z-score ≥ 1.5σ: above average, worth monitoring
- **Severe** — z-score ≥ 2.5σ: significant heat event, schedule follow-up
- **Critical** — z-score ≥ 3.5σ: extreme event, immediate attention recommended

### How to use it

**Filters at the top:**
- **Site** — narrow to one of the 8 sites or view all.
- **Severity** — filter by Warning, Severe, or Critical.
- **Status** — filter by New (current year, just detected) or Reviewed (all prior years).
- **Date range** — pick a time window to focus on.

**Summary strip:**
- Shows total anomalies, Critical and Severe counts, how many are New (current year), and the average and peak z-score for the current filter selection.

**Anomaly table (left):**
- Lists every detected event with date, site, observed LST, historical baseline LST, z-score, severity badge, status, and NDVI change (a negative NDVI change alongside a high LST often indicates vegetation stress).
- Shows up to 80 events at a time.

**Drill-down selector (below the table):**
- Pick any event from the dropdown to load its full explanation in the panel on the right.

**Event Details panel (right):**
- Shows the key numbers for the selected event and a plain-English narrative explaining what the anomaly means ecologically and what action is recommended.

**LST Time-Series chart (bottom):**
- Plots the full monthly LST record for the selected site with coloured markers for anomaly events and shaded bands for ±1σ and ±2σ historical ranges.
- The selected event is highlighted with a star marker.

### What you can learn from it
- Which sites and which seasons are most prone to heat anomalies.
- Whether anomaly frequency is increasing over time.
- How strongly LST and NDVI move together during heat events (vegetation stress signal).
- The worst heat events on record for each site going back to 2000.

### Live data note
Historical anomalies (2000–May 2026) are loaded from pre-processed files. Any months not yet in the dataset are fetched live from Google Earth Engine on page load, scored against the 2000–2015 baseline, and added to the table automatically. If a live anomaly is detected a banner appears at the top of the page.

### Status logic
- **Reviewed** — all anomalies from years before the current year. These are known historical events.
- **New** — anomalies detected in the current calendar year (including live GEE fetches). These are the ones worth acting on.

### Data note
158 real anomalies are detected across 2000–2023 from real MODIS satellite data, including 1 Critical and 16 Severe events. Years 2024–2025 show no anomalies — those years had above-baseline temperatures but did not cross the 1.5σ detection threshold at these 8 sites.
