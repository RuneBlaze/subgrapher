from quartet_extracter import QuartetExtractor
import treeswift as ts

tree = ts.read_tree_newick("(((A:1,B:1)0.66:1)0.33:1,C:1,D:1);")
extractor = QuartetExtractor(tree)
print(extractor.extract(("A", "B", "C", "D")))
