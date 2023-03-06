import treeswift as ts
from dataclasses import dataclass
from typing import Tuple, Optional
NORMALIZER = 0
import numpy as np
from copy import copy
from itertools import combinations

IGNORE_FULL_SUPPORT = True

def is_numeric(s : str):
    try:
        float(s)
        return True
    except ValueError:
        return False

@dataclass
class LogDomain:
    value: Optional[float] = 0
    num_zeros: int = 0

    def times(self, x):
        x = max(x, 0)
        if x > 0:
            self.value += np.log(x)
        else:
            self.num_zeros += 1

    def __truediv__(self, x):
        return LogDomain(self.value / x, self.num_zeros // x)
    
    def __add__(self, x):
        return LogDomain(self.value + x.value, self.num_zeros + x.num_zeros)
    
    def __sub__(self, x):
        return LogDomain(self.value - x.value, self.num_zeros - x.num_zeros)

    def consume(self, l):
        for x in l:
            self.times(x)

    @staticmethod
    def from_array(a):
        x = LogDomain()
        x.consume(a)
        return x

    def divide(self, x):
        x = max(x, 0)
        if x != 0:
            self.value -= np.log(x)
        else:
            self.num_zeros -= 1
            if self.num_zeros < 0:
                raise ValueError("Divide by zero")
            
    def divide_by(self, x : "LogDomain") -> "LogDomain":
        return LogDomain(self.value - x.value, self.num_zeros - x.num_zeros)
    
    def times_by(self, x : "LogDomain") -> "LogDomain":
        return LogDomain(self.value + x.value, self.num_zeros + x.num_zeros)

    def item(self):
        if self.value is None:
            return 1
        elif self.num_zeros > 0:
            return 0
        else:
            return np.exp(self.value)

def support(node):
    if node.is_leaf():
        return 1
    if node.label is None:
        return 1
    elif is_numeric(node.label):
        return max((float(node.label) - NORMALIZER), 0) / (1 - NORMALIZER)
    else:
        return 0

def supports_from_root(self):
    d = {}
    for node in self.traverse_preorder():
        if node.is_root():
            d[node] = LogDomain()
        else:
            d[node] = copy(d[node.parent])
            if not (node.is_leaf() and not IGNORE_FULL_SUPPORT):
                d[node].times(1 - support(node))
    return d

def hops_from_root(self):
    d = {}
    for node in self.traverse_preorder():
        if node.is_root():
            d[node] = 0
        else:
            d[node] = d[node.parent] + 1
    return d

def edge_lengths_from_root(self):
    d = {}
    for node in self.traverse_preorder():
        if node.is_root():
            d[node] = 0
        else:
            d[node] = d[node.parent] + node.edge_length
    return d

@dataclass
class Quartet:
    taxa : Tuple[str, str, str, str]
    order : Tuple[int, int, int, int]
    error_prob : Tuple[float, float, float, float, float]
    edge_length : Tuple[float, float, float, float, float]

    def taxa_in_order(self):
        return tuple(self.taxa[i] for i in self.order)
    
    def classify(self):
        return {
            (0, 1, 2, 3): 0,
            (0, 3, 1, 2): 1,
            (0, 2, 1, 3): 2,
        }[self.order]


from collections import defaultdict
def mrca_matrix(tree):
    '''Return a dictionary storing all pairwise MRCAs. ``M[u][v]`` = MRCA of nodes ``u`` and ``v``. Excludes ``M[u][u]`` because MRCA of node and itself is itself

    Returns:
        ``dict``: ``M[u][v]`` = MRCA of nodes ``u`` and ``v``
    '''
    descendants = defaultdict(list)
    M = defaultdict(dict)
    for node in tree.traverse_postorder():
        M[node][node] = node
        if node.is_leaf():
            pass
        else:
            for child in node.children:
                descendants[node].extend(descendants[child])
        for c in node.children:
            for d in descendants[c]:
                M[node][d] = node
                M[d][node] = node
        for branch1, branch2 in combinations(node.children, 2):
            for u in descendants[branch1]:
                for v in descendants[branch2]:
                    M[u][v] = node
                    M[v][u] = node
        descendants[node].append(node)
    return M

def solve(ab, cd, ac, bd, ad, bc):
    mid = (ac + bd - ab - cd) / 2
    _1_3 = ac - mid
    _1_4 = ad - mid
    _2_4 = bd - mid
    _1 = (ab + _1_4 - _2_4) / 2
    _2 = ab - _1
    _3 = _1_3 - _1
    _4 = _1_4 - _1
    return _1, _2, _3, _4, mid

class QuartetExtractor:
    def __init__(self, tree : ts.Tree):
        self.tree = tree
        self.mrca = mrca_matrix(tree)
        self.lookup = {l.label: l for l in tree.traverse_leaves()}
        self.depths = hops_from_root(tree)
        self.edge_lengths = edge_lengths_from_root(tree)
        self.supports = supports_from_root(tree)

    def error_prob_between(self, x, y):
        i = self.mrca[x][y]
        return self.supports[x].times_by(self.supports[y]).divide_by(self.supports[i]).divide_by(self.supports[i])
    
    def edge_length_between(self, x, y):
        i = self.mrca[x][y]
        return self.edge_lengths[x] + self.edge_lengths[y] - 2 * self.edge_lengths[i]
    
    def hop_distance_between(self, x, y):
        i = self.mrca[x][y]
        return self.depths[x] + self.depths[y] - 2 * self.depths[i]
    
    def extract_values(self, f, a, b, c, d):
        ab = f(a, b)
        cd = f(c, d)
        ac = f(a, c)
        bd = f(b, d)
        ad = f(a, d)
        bc = f(b, c)
        return solve(ab, cd, ac, bd, ad, bc)

    def extract(self, taxa : Tuple[str, str, str, str]):
        d_ab = self.hop_distance_between(self.lookup[taxa[0]], self.lookup[taxa[1]])
        d_cd = self.hop_distance_between(self.lookup[taxa[2]], self.lookup[taxa[3]])
        d_ad = self.hop_distance_between(self.lookup[taxa[0]], self.lookup[taxa[3]])
        d_bc = self.hop_distance_between(self.lookup[taxa[1]], self.lookup[taxa[2]])
        d_ac = self.hop_distance_between(self.lookup[taxa[0]], self.lookup[taxa[2]])
        d_bd = self.hop_distance_between(self.lookup[taxa[1]], self.lookup[taxa[3]])
        d1 = (d_ab + d_cd, (0, 1, 2, 3))
        d2 = (d_ad + d_bc, (0, 3, 1, 2))
        d3 = (d_ac + d_bd, (0, 2, 1, 3))
        min_d, order = min([d1, d2, d3], key=lambda x: x[0])
        x, y, a, b = [taxa[i] for i in order]
        x, y, a, b = self.lookup[x], self.lookup[y], self.lookup[a], self.lookup[b]
        supports = tuple(x.item() for x in self.extract_values(self.error_prob_between, x, y, a, b))
        edge_lengths = self.extract_values(self.edge_length_between, x, y, a, b)
        return Quartet(taxa, order, supports, edge_lengths)