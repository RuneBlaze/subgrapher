from subgrapher.quintet_counter import extract_clades
import treeswift as ts

def test_extract_clades():
    quintet_tree = ts.read_tree_newick('((A:1,B:1):1,((C:1,D:1):1,E:1));')
    clades = extract_clades(quintet_tree, ['A', 'B', 'C', 'D', 'E'])
    assert clades[2] == clades[3] == clades[18] == 1