"""Peeling optimization experiments."""
import random as rd

from pangesim.panevolve import PangenomeSimulator
from pangesim.reconstruction import EulerianPathHeuristic


def peeling():
    """Peeling experiment for optimization."""
    #Generating a ground truth graph
    sim = PangenomeSimulator(deletion_rate=1, rearrangement_rate=5)
    genomes = rd.randint(2, 5)
    genes = rd.randint(2, 15)
    g_truth = sim.generate_pangenome(k=genomes, length=genes)
    print("\t Generated ground truth: ")
    print(g_truth.summary())
    h = g_truth.compute_weighted_adjacencies()

    #Build base pangenome
    heuristic = EulerianPathHeuristic()
    pangenome = heuristic.reconstruct(h)
    print("\t Inferred pangenome: ")
    print(pangenome.summary())

    print("\t Pangenome length difference with ground truth: ", len(g_truth) - len(pangenome))



if __name__ == "__main__":
    peeling()
