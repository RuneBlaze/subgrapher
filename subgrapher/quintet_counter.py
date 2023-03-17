import argparse
import treeswift as ts
from table_five import TreeSet
import numpy as np
import torch
from itertools import combinations
from tqdm import tqdm


def remove_root_prefix(t):
    newick = t.newick()
    if newick[0] == "[":
        newick = newick[(newick.index("]") + 1) :]
    return newick


def sample_combinations(l, k, samples=100000):
    # sample "samples" # of  k combinations from l
    res = set()
    for _ in range(samples):
        idx = np.random.choice(len(l), k, replace=False)
        res.add(tuple(sorted(idx)))
    return [tuple(l[i] for i in c) for c in res]


def basis(a):
    r = [0] * 25
    for i in a:
        r[i] = 1
    return r


taxa = [0, 1, 2, 3, 4]
possible_clades = []
for k in [2, 3, 4]:
    for subset in combinations(taxa, k):
        candidate = [0] * 5
        for s in subset:
            candidate[s] = 1
        possible_clades.append(candidate)
possible_clades.sort()

clades2idx = {}
for i, e in enumerate(possible_clades):
    clades2idx[frozenset(zero_idx for zero_idx, it in enumerate(e) if it == 1)] = i


def extract_clades(tree, taxa):
    subtree = tree.extract_tree_with(taxa)
    clades = []
    for c in subtree.traverse_postorder():
        if c.is_leaf():
            c.clade = set([c.label])
        else:
            c.clade = set()
            for cc in c.children:
                c.clade = c.clade.union(cc.clade)
            if len(c.clade) < 5:
                clades.append(c.clade)
    # convert to numeric_indices
    assert len(clades) == 3
    return basis(
        [clades2idx[frozenset([taxa.index(l) for l in c])] for c in clades]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # reads in a directory name
    parser.add_argument("directory", help="directory to process")
    args = parser.parse_args()
    true_gene_trees = TreeSet(args.directory + "/truegenetrees")
    est_gene_trees = []
    true_species_tree = ts.read_tree_newick(args.directory + "/s_tree.trees")
    for i in range(1000):
        est_gene_trees.append(
            ts.read_tree_newick(args.directory + f"/gtrees_400.tre.l{i}.abayes.cleaned")
        )
    with open(args.directory + "/est_gene_trees", "w+") as f:
        # write out est_gene_trees
        for t in est_gene_trees:
            f.write(remove_root_prefix(t) + "\n")
    est_gene_trees = TreeSet(args.directory + "/est_gene_trees")
    labels = [n.label for n in true_species_tree.traverse_leaves()]
    dist_true = []
    dist_est = []
    topology_rooted = []
    quintets = sample_combinations(labels, 5)
    for q in tqdm(quintets):
        t1 = true_gene_trees.tally_single_quintet(q)
        t2 = est_gene_trees.tally_single_quintet(q)
        if sum(t1) == 0 or sum(t2) == 0:
            continue
        dist_true.append(np.asarray(t1, dtype=np.single) / sum(t1))
        dist_est.append(np.asarray(t2, dtype=np.single) / sum(t2))
        topology_rooted.append(extract_clades(true_species_tree, q))
        if len(dist_true) >= 50000:
            break
    dist_true = torch.tensor(dist_true).float()
    dist_est = torch.tensor(dist_est).float()
    topology_rooted = torch.tensor(topology_rooted).float()
    torch.save(dist_true, args.directory + "/dist_true.pt")
    torch.save(dist_est, args.directory + "/dist_est.pt")
    torch.save(topology_rooted, args.directory + "/topology_rooted.pt")
