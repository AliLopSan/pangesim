"""Pangesim: A discrete graph-based pangenome simulator.

This package provides discrete simulation frameworks to model pangenome structural
evolution, alongside combinatorial metrics to benchmark graph-reconstruction
heuristics and discrete optimization algorithms.
"""

from pangesim.datastructures import Genome
from pangesim.datastructures import Pangenome
from pangesim.panevolve import PangenomeSimulator

__version__ = "0.1.0"

__all__ = [
    "Genome",
    "Pangenome",
    "PangenomeSimulator",
]
