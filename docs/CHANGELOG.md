# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### 🌟 Features
- Added `scripts/run_example.py`, an interactive CLI smoke test and prototyping sandbox designed for undergrad students to validate evolutionary simulations.
- Added `pangesim.visualization` subpackage featuring the `PangenomeVisualizer` engine for matrix and layout rendering.
- Implemented `LinearSimulator` for tree-less pangenome evolution with stochastic complexity tiers ("low", "medium", "high").
- Added safe gene deletion logic (`_apply_gene_deletions`) adhering to the minimum 2-gene constraint per genome and path.
- Introduced `simulation_history` tracking and `print_simulation_summary()` for experiment provenance reporting.
### 📦 Build
- Optimized visualization testing routines to utilize secure temporary file directories (`tmp_path`) protecting local environments from artifact pollution.
