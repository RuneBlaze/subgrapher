import treeswift as ts

def bipartitions(t : ts.Tree):
    leaves = frozenset(n.label for n in t.traverse_leaves())
    bipartitions = []
    for n in t.traverse_postorder():
        if n.is_leaf():
            n.clades = frozenset([n.label])
            continue
        else:
            if n.is_root():
                continue
            buf = []
            for c in n.children:
                buf.extend(c.clades)
            n.clades = frozenset(sorted(buf))
            a, b = sorted([n.clades, leaves - n.clades], key = lambda x: list(x))
            bipartitions.append((a, b))
    return frozenset(bipartitions)

def topology_agree(g_star, g, t):
    subtree1 = g_star.extract_tree_with(t, suppress_unifurcations=True)
    # print(subtree1.newick())
    subtree2 = g.extract_tree_with(t, suppress_unifurcations=True)
    # print(subtree2.newick())
    return bipartitions(subtree1) == bipartitions(subtree2)