# Pangesim

### Reconstructing genome sets from adjacency information using combinatorial optimization

------

## Overview

Pangenomes are commonly represented as sets of genomes sharing a subset of genes and genomic adjacencies.

In many situations, however, only  adjacency information (sequencing reads, synteny blocks) is available:

- Which gene adjacencies exist?
- How frequently does each adjacency appear?
- Which genes are conserved across all genomes?

Recovering the original set of genomes from this compressed representation is not straightforward.

This project investigates the reconstruction of pangenomes and its constituent genomes from adjacency data alone.

The repository contains:

- A simulator of pangenome evolution
- Synthetic benchmark generation
- Reconstruction heuristics
- Evaluation metrics
- Experimental notebooks

------

## Why is this problem interesting?

A pangenome can be viewed as a collection of genome paths. Different genome collections may induce very similar adjacency frequencies. This creates a combinatorial reconstruction problem:

> Given only adjacency information, can we recover the underlying genomes?

The figure below illustrates the challenge:

![problem_definition](./docs/images/problem_definition.png)

Our input (on the left) is a graph that represents the known adjacencies between the genes. The weight of its edges represents the number of times that this adjacency was reported on the data. Our goal is to find a  Pangenome $$\mathcal{P}$$ that could explain this data. However, there could exist more than one explanation, as we can see on the right: there are two pangenomes that could explain the graph on the left.

------

## Project Components

### 1. Pangenome Simulator

Starting from an ancestral genome and a species tree, the simulator generates synthetic pangenomes through evolutionary events such as:

- Gene loss
- Gene insertion
- Genome rearrangements

The simulator produces a complete ground truth:

- Genome paths
- Edge frequencies
- Core genes
- Evolutionary history

------

### 2. Reconstruction Heuristic

The reconstruction algorithm attempts to infer:

- The number of genomes
- Genome structures
- Edge multiplicities
- Core genome composition

using only adjacency information.

------

### 3. Evaluation Framework

The simulator provides a known ground truth, allowing quantitative assessment of reconstruction quality.

Metrics include:

- Genome count accuracy
- Core genome recovery
- Edge precision and recall

## Repository Structure

```text
.
├── src/pangesim/
│   ├── analysis/
│   ├── io/
│   ├── datastructures/
│   ├── panevolve/ #main simulator
│   ├── reconstruction/ #algorithms and heuristics
│   ├── tools/
│   └── visualization/
│
├── script/
│   ├── run_benchmark.py
│   └── run_results.py
│
├── docs/
│   ├── images/
│   ├── gifs/
│   └── figures/
│
├── data/
│
└── README.md
```

------

## Quick start guide

This project uses `pygraphviz` and `LaTeX` for high-fidelity pangenome network visualizations. Because these depend on system-level binaries, you must install them on your machine before setting up the Python environment.

### 1. Install System Dependencies
#### 🍏 macOS (via Homebrew)
```bash
brew install graphviz mactex-no-gui
```
#### 🐧 Ubuntu/Linux
```bash
sudo apt-get update
sudo apt-get install graphviz graphviz-dev texlive-full
```

### 2. Initialize Python Environment
Once system binaries are installed, let `uv` handle the rest:
```bash
uv sync --all-groups
```


We provide a built-in smoke test script to quickly verify your installation, run a toy pangenome evolution simulation, and output a structural grid visualization.

### Running the sample simulation 

You can execute the example script directly using `uv`:

```bash
uv run python ./scripts/run_example.py 
```

The resulting output should look like this:

![Pangenome Grid Output](docs/images/pangenome_example.png)

------

## Research Questions

This repository explores questions such as:

- How much information is lost when only adjacency data is available?
- Which adjacency graph properties make reconstruction difficult or non-unique?
- How accurately can endogenous edge frequencies be recovered?
- How does evolutionary complexity affect reconstruction quality?

------

## Status

Current stage:

-  Pangenome simulator (In progress)
-  Ground-truth generation (In progress)
-  Reconstruction heuristic benchmarking (In progress)
-  Comparative baselines (Future work)
-  Large-scale experiments (Future work)



## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup and coding standards guidelines.
