import networkx as nx
import numpy as np
import os
import random

def get_dir_from_path_list(path):
    outdir = path[0]
    for p in path[1:]:
        outdir = os.path.join(outdir, p)
        os.makedirs(outdir, exist_ok=True)
    return outdir

def custom_watts_strogatz_graph(n, k, p, seed=random):
    p, node_names = p

    if k >= n:
        print("K: ", k, " N: ", n)
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.Graph()
    nodes = node_names
    for j in range(1, k // 2 + 1):
        assert p != 0 and k != 0
        targets = nodes[j:] + nodes[0:j]
        G.add_edges_from(zip(nodes, targets))

    for j in range(1, k // 2 + 1):
        assert p != 0 and k != 0
        targets = nodes[j:] + nodes[0:j]

        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)

                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G

def normal_watts_strogatz_graph(n, agents, mu, sigma, seed=random):
    if mu >= n:
        print("mu: ", mu, " N: ", n)
        raise nx.NetworkXError("mu>=n, choose smaller mu or larger n")

    G = nx.Graph()
    nodes = agents

    degrees = np.random.normal(loc=mu, scale=sigma, size=n).astype(int)

    degrees = np.clip(degrees, 1, n - 1)

    for i, node in enumerate(nodes):
        degree = degrees[i]
        targets = nodes[i+1:i+1+degree] + nodes[:max(0, (i+1+degree) - len(nodes))]
        G.add_edges_from(zip([node] * len(targets), targets))

    return G
