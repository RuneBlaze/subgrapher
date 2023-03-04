import argparse
import treeswift as ts
from itertools import combinations
import numpy as np
from extract_subgraph import create_fake_data
import torch
from tqdm import tqdm

def sample_combinations(l, k, samples = 5000):
    # sample "samples" # of  k combinations from l
    res = set()
    for _ in range(samples):
        idx = np.random.choice(len(l), k, replace=False)
        res.add(tuple(sorted(idx)))
    return [tuple(l[i] for i in c) for c in res]

def read_tree_list(filepath):
    trees = []
    with open(filepath, 'r') as f:
        for line in f:
            trees.append(ts.read_tree_newick(line))
    return trees

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # reads in a directory name
    parser.add_argument('directory', help='directory to process')
    args = parser.parse_args()
    # read trees in the form of gtrees_400.tre.l325.nni.abayes.cleaned
    true_gene_trees = read_tree_list(args.directory + '/truegenetrees')
    est_gene_trees = []
    for i in tqdm(range(1000)):
        est_gene_trees.append(ts.read_tree_newick(args.directory + f'/gtrees_400.tre.l{i}.nni.abayes.cleaned'))
    labels = [n.label for n in true_gene_trees[0].traverse_leaves()]
    quintets = sample_combinations(labels, 5)
    quartets = sample_combinations(labels, 4)
    datasets = []
    for i in tqdm(range(1000)):
        for q in quintets:
            datasets.append(create_fake_data(est_gene_trees[i], true_gene_trees[i], q))
    # for i in range(1000):
    #     datasets.append(create_fake_data(est_gene_trees[i], true_gene_trees[i], quartets))
    torch.save(datasets, args.directory + '/datasets.pt')