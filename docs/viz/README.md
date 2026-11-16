# `src/firm_ce/viz/` — Layer 6: Visualisation & Plotting Suite

## Summary
The `firm_ce.viz` subpackage houses all Matplotlib, GeoPandas, and Seaborn visualisation tools for FIRM_CE. Separating plotting into `viz/` keeps [`firm_ce.system`](./docs/system/README.md) and [`firm_ce.analysis`](./docs/analysis/README.md) headless and lightweight.

---

## Module Breakdown

### 1. [`style.py`](./src/firm_ce/viz/style.py)
- **`TECH_COLORS` & `TECH_HATCHES`**: Canonical colour palette and hatch patterns for all technologies, sub-technologies, composite groups, and balance terms (`Curtailment`, `Storage Losses`, `Transmission Losses`, `Spillage`).
- **`get_color(tech)` & `get_hatch(tech)`**: Resolves colours and hatches (including `"//"` for electrical charging and `".."` for inflows).
- **`class PlotStyle`**: Scoped Matplotlib `rc_context` configuration and significant-figure formatter (`format_sigfigs`).

### 2. [`presets.py`](./src/firm_ce/viz/presets.py)
- **`class MetricSpec`**: Declarative specification for aggregating and normalising asset metrics.
- **`MAP_SPECS` & `SUMMARY_PRESETS`**: Pre-configured specs for spatial maps (`capacity`, `new_capacity`, `energy`, `net_energy`, `line_capacity`, `line_energy`) and fleet summary charts (`system_overview`, `power_capacity`, `energy_balance`, `storage_profile`, `system_overview_with_losses`).

### 3. [`display.py`](./src/firm_ce/viz/display.py)
- **`class Display`**: Spatial and fleet-level visualiser for single solutions and MHMGA ensembles:
  - `plot_map(...)`: European GeoJSON network maps (`EPSG:3035`) with scaled nodal pies or bars, transmission corridors, multi-panel atlases, and delta comparisons.
  - `plot_summary(...)` & `plot_single_balance_bars(...)`: Stacked bars (with negative charging/loss/curtailment stacks), recursive bisection treemaps, and area-scaled pie charts.

### 4. [`behaviour_plots.py`](./src/firm_ce/viz/behaviour_plots.py)
- **`class Behaviour(BehaviourAnalyzer)`**: Extends `BehaviourAnalyzer` with Matplotlib plotting methods:
  - `plot_daily_average(...)`: Multi-period diurnal stacked area or line profiles.
  - `plot_raw_trace(...)`: Unaveraged chronological dispatch/SOC traces over any interval window.
  - `plot_storage_charging_sources(...)` & `plot_storage_cross_charging(...)`: Diurnal profiles, raw traces, 2D GWh/yr heatmaps, and stacked bars of storage charging sources.
  - `plot_capacity_factors(...)`: 3-panel operational diagnostics dashboard (run-hours/yr & CF bars, diurnal duty cycle, and conditional output while online vs overall average).

### 5. [`space_explorer.py`](./src/firm_ce/viz/space_explorer.py)
- **`class SpaceExplorer`**: High-dimensional near-optimal ensemble visualiser built on `EnsembleDataset`:
  - `plot_surface(...)`: 2D binned/quantized cost or capacity surfaces with LCOE isolines.
  - `plot_pairgrid(...)`: Pairwise technology trade-off matrix with optional 2D Pareto frontiers and diagonal histograms.
  - `plot_slack_ribbons(...)`: Min-max and inter-quantile capacity envelopes as a function of LCOE relaxation (%).
  - `plot_distributions(...)`: 1D marginal boxen, violin, or box plots.

### 6. [`uncertainty_plots.py`](./src/firm_ce/viz/uncertainty_plots.py)
- **`plot_lcoe_distributions`**: Kernel density estimates of LCOE distributions + baseline LCOE number line under cost uncertainty.
- **`plot_rank_probability`**: Heatmap of rank probabilities (`1st .. kth+`) across top alternatives.
- **`plot_pairwise_win_rates`**: Head-to-head win-rate probability matrix $P(\text{LCOE}_i < \text{LCOE}_j)$.
