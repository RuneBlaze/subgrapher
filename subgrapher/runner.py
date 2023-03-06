import argparse
import treeswift as ts
from itertools import combinations
import numpy as np
from extract_subgraph import create_fake_data
from quartet_extracter import QuartetExtractor
from dataclasses import asdict
import torch
from tqdm import tqdm

def sample_combinations(l, k, samples = 50000):
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
    true_gene_trees = read_tree_list(args.directory + '/truegenetrees')
    est_gene_trees = []
    for i in tqdm(range(1000)):
        est_gene_trees.append(ts.read_tree_newick(args.directory + f'/gtrees_400.tre.l{i}.abayes.cleaned'))
    labels = [n.label for n in true_gene_trees[0].traverse_leaves()]
    quartets = sample_combinations(labels, 4)
    datasets = []
    BASIS = [torch.tensor([1,0,0]), torch.tensor([0,1,0]), torch.tensor([0,0,1])]
    for i in tqdm(range(1000)):
        eT = est_gene_trees[i]
        tT = true_gene_trees[i]
        ext_eT = QuartetExtractor(eT)
        ext_tT = QuartetExtractor(tT)
        for q in quartets:
            try:
                d = ext_eT.extract(q)
                dp = {}
                dp['predicted_label'] = BASIS[d.classify()]
                dp['true_label'] = BASIS[ext_tT.extract(q).classify()]
                dp['order'] = torch.tensor(d.order)
                dp['error_prob'] = torch.tensor(d.error_prob).float()
                dp['edge_length'] = torch.tensor(d.edge_length).float()
                dp['taxa'] = d.taxa
                datasets.append(dp)
            except ValueError:
                pass
    torch.save(datasets, args.directory + '/datasets.pt')