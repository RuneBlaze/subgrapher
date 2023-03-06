import treeswift as ts
from subgrapher.quartet_extracter import QuartetExtractor
from itertools import combinations
import numpy as np

def test_can_extract_quartet():
    tree = ts.read_tree_newick("((A:1,B:1):1,C:1,D:1);")
    extractor = QuartetExtractor(tree)
    q = extractor.extract(("A", "B", "C", "D"))
    assert q.taxa == ("A", "B", "C", "D")
    assert q.order == (0, 1, 2, 3)
    assert q.error_prob == (0.0, 0.0, 0.0, 0.0, 0.0)

def test_mrca():
    tree = ts.read_tree_newick("res/n30.supp.tre")
    extractor = QuartetExtractor(tree)
    leaves = sorted(extractor.lookup.keys())
    for u, v in combinations(leaves, 2):
        u, v = extractor.lookup[u], extractor.lookup[v]
        assert extractor.mrca[u][v] == extractor.mrca[v][u]
        assert tree.mrca([u.label, v.label]) == extractor.mrca[u][v]

def test_distance_right():
    tree = ts.read_tree_newick("res/n30.supp.tre")
    D = tree.distance_matrix(True)
    extractor = QuartetExtractor(tree)
    leaves = sorted(extractor.lookup.keys())
    for u, v in combinations(leaves, 2):
        u, v = extractor.lookup[u], extractor.lookup[v]
        assert np.allclose(extractor.edge_length_between(u, v), D[u.label][v.label])

def test_can_calculate_edge_weights():
    tree = ts.read_tree_newick("res/n30.supp.tre")

    extractor = QuartetExtractor(tree)
    leaves = sorted(extractor.lookup.keys())
    D = tree.distance_matrix(True)
    for (x, y, a, b), _ in zip(combinations(leaves, 4), range(10000)):
        q = extractor.extract((x, y, a, b))
        _1,_2,_3,_4 = q.taxa_in_order()
        assert D[_1][_2] + D[_3][_4] < D[_1][_3] + D[_2][_4], tree.extract_tree_with([x, y, a, b]).newick()
        assert np.allclose(q.edge_length[0] + q.edge_length[4] + q.edge_length[2], D[_1][_3])