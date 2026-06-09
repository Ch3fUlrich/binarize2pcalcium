# binarize2pcalcium Documentation

Welcome to the documentation for `binarize2pcalcium` — a Python package for
binarizing continuous calcium imaging traces (both 2-photon and 1-photon).

## Documentation files

| File | Content |
|------|---------|
| [📘 refactoring.md](refactoring.md) | Overview of the refactoring from a monolithic 4,111-line class to 10 modular files. Explains why the change was made, the before/after structure, an API migration guide from the old `Calcium` class to the new `binarize()` function, and a list of 7 bugs fixed during the refactoring. |
| [📗 pipeline.md](pipeline.md) | Complete reference for the single entry point `binarize()` function and the `BinarizationResult` dataclass. Full parameter table with types, defaults, and descriptions. The exact 8-step pipeline flow diagram. Save/load methods. Usage with config files and simulated data. |
| [📙 modules.md](modules.md) | Comprehensive module reference covering every public function across all 10 modules (`filters.py`, `thresholds.py`, `preprocess.py`, `binarization.py`, `correlation.py`, `overlap.py`, `dedup.py`, `pca.py`, `io.py`, `data_simulation.py`). Each function has its full signature, parameter table, return type, and description. |
| [📕 config.md](config.md) | Configuration reference for the YAML pipeline files (`config/default.yaml`, `config/2p_pipeline.yaml`, `config/1p_pipeline.yaml`). Every parameter listed with its 2P and 1P default values, detailed explanation, and a parameter tuning guide for common scenarios (too many events, too few events, noisy data). |
| [📓 api.md](api.md) | Quick-start API cheatsheet. Primary API, module import examples, and 5 common usage patterns: custom thresholds, 1P binarization, config file usage, simulation testing, inter-cell correlations, and footprint overlap analysis. |

## Quick links

- **Primary API**: [`binarize(F, ...)` → `BinarizationResult`](pipeline.md)
- **Module reference**: [all 10 modules](modules.md)
- **Configuration**: [YAML parameter reference](config.md)
- **Migration guide**: [Calcium class → binarize() function](refactoring.md)
- **Examples**: [Jupyter notebooks](../notebooks/)
