import treeswift as ts
from typing import List
import networkx as nx
import numpy as np
from dataclasses import dataclass
from comparer import topology_agree
import torch
NORMALIZER = 0.3333
def ts2nx(t : ts.Tree):
    G = nx.Graph()
    node2id = {}
    for i, n in enumerate(t.traverse_postorder()):
        node2id[n] = i
        G.add_node(i)
    # add the edges
    for n in t.traverse_postorder():
        for c in n.children:
            if c.is_leaf():
                error_prob = 0
            else:
                error_prob = 1 - max((float(c.label) - NORMALIZER), 0) / (1 - NORMALIZER)
                error_prob = np.log(error_prob) if error_prob > 0 else -np.inf
            G.add_edge(node2id[n], node2id[c], support = error_prob, weight = c.edge_length, num_edges = 1)
    return G

def contract_node(g, u):
    # remove this degree 2 node, and add the edges
    assert g.degree(u) == 2
    v, w = list(g.neighbors(u))
    supp1 = g.get_edge_data(v, u)['support']
    supp2 = g.get_edge_data(u, w)['support']
    weight1 = g.get_edge_data(v, u)['weight']
    weight2 = g.get_edge_data(u, w)['weight']
    num_edges1 = g.get_edge_data(v, u)['num_edges']
    num_edges2 = g.get_edge_data(u, w)['num_edges']
    g.remove_node(u)
    g.add_edge(v, w, support = supp1 + supp2, weight = weight1 + weight2, num_edges = num_edges1 + num_edges2)

def extract_graph(t : ts.Tree, taxa : List[str]):
    subtree = t.extract_tree_with(taxa, suppress_unifurcations=False)
    print(subtree.newick())
    G = ts2nx(subtree)
    # contract the nodes
    while True:
        nodes = [n for n in G.nodes() if G.degree(n) == 2]
        if len(nodes) == 0:
            break
        for n in nodes:
            if G.degree(n) == 2:
                contract_node(G, n)
    for e in G.edges():
        oldsupport = np.exp(G[e[0]][e[1]]['support'])
        G[e[0]][e[1]]['support'] = 1 - oldsupport
        assert G[e[0]][e[1]]['support'] >= 0
    return G

@dataclass
class FakeData:
    x : torch.Tensor
    edge_index : torch.Tensor
    edge_attr : torch.Tensor
    y : torch.Tensor

def nx2pyg(G):
    node2id = {n : i for i, n in enumerate(G.nodes())}
    edge_index = torch.tensor([[node2id[e[0]], node2id[e[1]]] for e in G.edges()], dtype = torch.long).t().contiguous()
    edge_attr = torch.tensor([[G[e[0]][e[1]]['support'], G[e[0]][e[1]]['weight'], G[e[0]][e[1]]['num_edges']] for e in G.edges()])
    x = torch.tensor([[1] if G.degree(n) == 1 else [0] for n in G.nodes()])
    y = torch.tensor([[0]])
    return FakeData(x, edge_index, edge_attr, y)

def create_fake_data(est_tree, true_tree, q):
    G = extract_graph(est_tree, q)
    draft = nx2pyg(G)
    draft.y = torch.tensor([[1 if topology_agree(est_tree, true_tree, q) else 0]])
    print(draft)
    return draft