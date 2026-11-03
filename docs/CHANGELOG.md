# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### 🌟 Features
- Added `scripts/run_example.py`, an interactive CLI smoke test and prototyping sandbox designed for undergrad students to validate evolutionary simulations.
- Added `pangesim.visualization` subpackage featuring the `PangenomeVisualizer` engine for matrix and layout rendering.
- Implemented `LinearSimulator` for tree-less pangenome evolution with stochastic complexity tiers ("low", "medium", "high").
- Added safe gene deletion logic (`_apply_gene_deletions`) adhering to the minimum 2-gene constraint per genome and path.
- Introduced `simulation_history` tracking and `print_simulation_summary()` for experiment provenance reporting.
- - Added `MwmComparisonVisualizer` for comparative dumbbell plots and analytical evaluation pipelines using Maximum Weight Matching (MWM) for pangenome single scenario evaluation.
- Added functions for $k$ results, Mean Absolute Percentage Error (MAPE), and adjusted publication figures for experimental analyses.
### 📦 Build
- Optimized visualization testing routines to utilize secure temporary file directories (`tmp_path`) protecting local environments from artifact pollution.
### 🐛 Bug fixes
- Refactored `Genome.add_edge` for robust path merging and traversal, accounting for genome modifications based on whether involved nodes are heads or tails, and integrating integer sets for constituent path lookups in $O(1)$ time.
- Replaced recursive postorder traversal with iterative traversal in MST handling.
- Harmonized node unwrapping for mixed primitive and container types.

### ♻️ Refactorings
- Standardized imports, output paths, and Ruff formatting/import orders across modules.

### 📚 Documentation
- Updated manuscript figures and styling adjustments (RECOMB publication standards).