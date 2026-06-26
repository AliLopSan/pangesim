"""Classification metrics."""

import logging
from typing import Dict

from pangesim import Pangenome
from pangesim.metrics.base import PangenomeMetric

# Standard internal logger for pipeline monitoring
logger = logging.getLogger(__name__)


class CoreContingencyTable(PangenomeMetric):
    """Calculates classification performance w.r.t core genes."""

    @property
    def name(self) -> str:
        """Prints metric name."""
        return "Core Contingency Table"

    def evaluate(self, ground_truth: Pangenome, inferred: Pangenome) -> Dict[str, float]:
        """Computes contingency table for core genes.

        Args:
            ground_truth: The benchmark pangenome.
            inferred: The pangenome whose properties are compared against the 'truth'.

        Returns:
            A dictionary containing performance metrics.
        """
        true_core = ground_truth.compute_core_genes()
        inf_core = inferred.compute_core_genes()

        tp, fp, fn, tn = 0, 0, 0, 0

        if len(inf_core) > 0:
            for g in inf_core:
                if g in true_core:
                    tp += 1
                else:
                    fp += 1

        for g in true_core:
            if g not in inf_core:
                fn += 1
        if len(inf_core) == 0:
            tn = len(inferred.universal_gene_set)
        else:
            for g in inferred.universal_gene_set:
                if g not in inf_core:
                    tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else float("nan")
        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }


class EdgeContingencyTable(PangenomeMetric):
    """Calculates classification performance w.r.t core genes."""

    @property
    def name(self) -> str:
        """Prints metric name."""
        return "Edge Contingency Table"

    def evaluate(self, ground_truth: Pangenome, inferred: Pangenome) -> Dict[str, float]:
        """Computes contingency table for core genes.

        If the universal gene sets fo not match, calculating edge classification
        if mathematically undefined. In this case entries are logged and
        float("nan") values are returned to preserve execution continuity.

        Args:
            ground_truth: The benchmark pangenome.
            inferred: The pangenome whose properties are compared against the 'truth'.

        Returns:
            A dictionary containing the true positives, false positives, etc.
        """
        if ground_truth.universal_gene_set != inferred.universal_gene_set:
            mismatched_genes = ground_truth.universal_gene_set ^ inferred.universal_gene_set
            mismatch_size = len(mismatched_genes)
            gt_size = len(ground_truth.universal_gene_set)
            mismatch_pct = (mismatch_size / gt_size) * 100 if gt_size > 0 else 0.0
            logger.warning(
                "[Metrics] Universal gene set mismatch discovered. "
                "Symmetric difference: %d genes (%.2f%% of ground truth's genes)"
                "Skipping edge contingency table computation.",
                mismatch_size,
                mismatch_pct,
            )
            return {
                "tp": float("nan"),
                "fp": float("nan"),
                "fn": float("nan"),
                "tn": float("nan"),
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
        true_adjacencies = ground_truth.compute_weighted_adjacencies()
        pred_adjacencies = inferred.compute_weighted_adjacencies()

        true_edges = set(true_adjacencies)
        pred_edges = set(pred_adjacencies)

        tp, fp, fn, tn = 0, 0, 0, 0

        for edge in pred_edges:
            if edge in true_edges:
                tp += 1
            else:
                fp += 1
        for edge in true_edges:
            if edge not in pred_edges:
                fn += 1

        order = len(inferred.universal_gene_set)
        tn = (order - (order - 1) // 2) - (tp + fp + fn)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else float("nan")
        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
