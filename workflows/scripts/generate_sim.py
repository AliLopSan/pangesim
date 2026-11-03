import random as rd
from typing import Tuple

import numpy as np

from pangesim.panevolve import LinearSimulator


def generate_pan(num_genes: int, evol: str) -> Tuple:
    """Returns a random simulated pangenome.

    Args:
    num_genes: Number of desired genes.
    evol: Complexity of evolutionary events "low", "medium", "high".

    Raises:
       TypeError: If num_genes is not an integer.
       ValueError: If num_genes is less than 2 or evol not in options.
    """
    num_genomes = rd.randint(10,30)
    rates = (0.0, 5.0)
    r1 = np.random.uniform(rates[0], rates[1])
    r2 = np.random.uniform(rates[0], rates[1])
    r3 = np.random.uniform(rates[0], rates[1])
    r4 = np.random.uniform(rates[0], rates[1])
    r5 = np.random.uniform(rates[0], rates[1])
    r6 = np.random.uniform(rates[0], rates[1])
    r7 = np.random.uniform(rates[0], rates[1])
    r8 = np.random.uniform(rates[0], rates[1])

    # Tests for the different complexity levels
    sim = LinearSimulator(
        insertion_rate=r1,
        deletion_rate=r2,
        rearrangement_rate=r3,
        inversion_rate=r4,
        fission_rate=r5,
        fusion_rate=r6,
        translocation_rate=r7,
        gene_deletion_rate=r8,
    )

    ground_truth = sim.generate_pangenome(k=num_genomes, length=num_genes, complexity=evol)

    return sim.simulation_history, ground_truth
