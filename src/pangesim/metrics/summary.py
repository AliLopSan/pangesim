"""Global inference metrics."""

from typing import Dict

import pandas as pd

from pangesim import Pangenome
from pangesim.metrics.base import Metric
from pangesim.metrics.base import PangenomeMetric


class PangenomeSummaryMetrics(PangenomeMetric):
    """Calculates macroscopic delta metrics between two pangenomes."""

    @property
    def name(self) -> str:
        """Prints metric name."""
        return "Pangenome Summary Metrics"

    def evaluate(self, ground_truth: Pangenome, inferred: Pangenome) -> Dict[str, float]:
        """Computes number of inferred genomes, core size and score.

        Args:
        ground_truth: The benchmark pangenome.
        inferred: The reconstructed pangenome.

        Returns:
        A dictionary with the global metrics.
        """
        k = len(ground_truth) - len(inferred)
        c = len(ground_truth.core) - len(inferred.core)

        return {"k_diff": abs(k), "c_diff": abs(c)}


class PangenomeDistribution(Metric):
    """Calculates dataframes for paths and weights."""

    __slots__ = ("pangenome","name")

    def __init__(self, pangenome:Pangenome, name:str = "Pangenome") -> None:
        """Initializator for class.

        Args:
        pangenome: Pangenome Datastructure.
        name: Optional label identifier.
        """
        self.pangenome = pangenome
        self.name = name

    def get_genome_summary(self)->pd.DataFrame:
        """Returns per genome level metrics: number of genes and path length."""
        records = []

        for genome in self.pangenome._genomes:
            path_id = 0
            for path in genome.get_path_sequences():
                records.append({"pan":self.name,
                                "genome id":genome._genome_id,
                                "genome size":len(genome.gene_set),
                                "path_id":path_id,
                                "length":len(path)})
                path_id += 1
        return pd.DataFrame(records)

    def get_weight_distribution(self) -> pd.DataFrame:
        """Returns edge weights across the associated weighted adjacency graph."""
        w = self.pangenome.compute_weighted_adjacencies()

        records = []

        for u,v in w:
            records.append({"u":u,"v":v,"weight":w[(u,v)]})

        return pd.DataFrame(records)
