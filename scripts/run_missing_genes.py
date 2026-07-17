from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.refining import ResidualsRefinement
from pangesim.reconstruction.sorting import WeightSorting


def sample_pangenome():
    """Returns a stable 5-genome mock pangenome."""
    # Ground-truth genomes
    gt_g1 = Genome(genome_id=0)
    g1_edges = [(1, 2), (2, 6), (6, 7), (7, 9), (9, 11), (3, 10), (3, 4), (4, 5)]
    for edge in g1_edges:
        gt_g1.add_edge(edge)

    gt_g2 = Genome(genome_id=1)
    g2_edges = [(2, 3), (3, 10), (5, 4), (4, 8)]
    for edge in g2_edges:
        gt_g2.add_edge(edge)

    gt_g3 = Genome(genome_id=2)
    g3_edges = [(2, 3), (3, 10)]
    for edge in g3_edges:
        gt_g3.add_edge(edge)

    gt_g4 = Genome(genome_id=3)
    g4_edges = [(1, 2), (2, 3), (3, 4), (4, 8), (10, 9), (9, 11)]
    for edge in g4_edges:
        gt_g4.add_edge(edge)

    gt_g5 = Genome(genome_id=4)
    g5_edges = [(1, 2), (2, 3), (3, 10), (10, 9), (5, 4), (4, 8)]
    for edge in g5_edges:
        gt_g5.add_edge(edge)

    ground_truth = Pangenome(pangenome_id="ground_truth")
    ground_truth.add_genome(gt_g1)
    ground_truth.add_genome(gt_g2)
    ground_truth.add_genome(gt_g3)
    ground_truth.add_genome(gt_g4)
    ground_truth.add_genome(gt_g5)

    return ground_truth

def investigate_gene_leak():
    """Investigates gene leak."""
    truth = sample_pangenome()
    matrix = truth.compute_weighted_adjacencies()
    b = GreedyPairingISCB()
    w = WeightSorting()
    a = EulerianTrailAssignment(trail_sorting=w)
    params = {"alpha":0.5, "gamma":1}
    k_min,k_max, k_info = b.compute_bounds(matrix, params)
    base_pangenome = a.assign_genomes(matrix, k_min)

    print(truth.summary())
    print(base_pangenome.summary())

    print("\t Gene set truth: ", truth.total_gene_count)
    print("\t Gene set base: ", base_pangenome.total_gene_count)

    r = ResidualsRefinement(params=params)
    refined_pangenome = r.refine(source=matrix, target=base_pangenome,
                                 ground_truth=truth)

    print("\n--After refinement ---")
    print(refined_pangenome.summary())

    print("\t Gene set truth: ", truth.total_gene_count)
    print("\t Gene set refined: ", refined_pangenome.total_gene_count)

    print("\nCHECKING evaluate condition: ")
    if truth.universal_gene_set != base_pangenome.universal_gene_set:
        print("Sets are not equal")
    else:
        print("Sets are equal")

    print("\t Truth: ", truth.universal_gene_set)
    print("\t Base: ", base_pangenome.universal_gene_set)

    if truth.universal_gene_set != refined_pangenome.universal_gene_set:
        print("Sets are not equal")
        print("\t Truth: ", truth.universal_gene_set)
        print("\t Refined: ", refined_pangenome.universal_gene_set)

if __name__ == "__main__":
    investigate_gene_leak()

