import dgl 
import os
import pickle
import networkx as nx
from scipy import sparse
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import ZIGZAG.zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d
import time
from scipy.stats import multivariate_normal
from ripser import Rips
from persim import PersistenceImager
import matplotlib.pyplot as plt
import re

data_name = 'DBLP_adj'
graph_index = 1
graph_class = 1
# data_dir = "data/DBLP"   # or "data/DBLP/data/DBLP"
if data_name == 'sbm':
    data_dir = f"data/sbm_m{graph_class}/"
    dataset = []

    # for i in range(5):
    #     path = os.path.join(data_dir, f"graph_{i}_by_edges")
    #     with open(path, "rb") as f:
    #         g_dgl = pickle.load(f)
    #     nx_g = dgl.to_networkx(g_dgl).to_undirected()
    #     nx_g = nx.Graph(nx_g) 
    #     dataset.append(nx_g)
    # print(dataset)

    graph_number = 100
    for index in range(graph_number - 1):
        mat = sparse.load_npz(data_dir + f"G{graph_index}_graph_{index}_adj.npz")
        nx_g = nx.from_scipy_sparse_array(mat)
        nx_g = nx_g.to_undirected() 
        nx_g = nx.Graph(nx_g) 
        dataset.append(nx_g)
        
    node_counts = [g.number_of_nodes() for g in dataset]
    NVertices = min(node_counts)

    aligned_dataset = []

    for g in dataset:
        nodes_sorted = sorted(g.nodes())
        nodes_keep = nodes_sorted[:NVertices]
        g_sub = g.subgraph(nodes_keep).copy()
        # g_sub = nx.convert_node_labels_to_integers(g_sub)
        g_relabel = nx.convert_node_labels_to_integers(g_sub, ordering='sorted')
        aligned_dataset.append(g_relabel)

    dataset = aligned_dataset

elif data_name == 'eth':
    data_dir = "data/ETH_continual_graphs"
    dataset = []

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith("_continual_graph.pt")
    )

    for f in files:
        path = os.path.join(data_dir, f)

        data = torch.load(path)

        # ---- case 1: already networkx graph ----
        if isinstance(data, nx.Graph):
            nx_g = data

        # ---- case 2: PyG style ----
        elif hasattr(data, "edge_index"):
            edge_index = data.edge_index.cpu().numpy()
            nx_g = nx.Graph()
            edges = list(zip(edge_index[0], edge_index[1]))
            nx_g.add_edges_from(edges)

        # ---- case 3: DGL graph ----
        elif "graph" in data:
            nx_g = dgl.to_networkx(data["graph"]).to_undirected()

        # ---- case 4: adjacency matrix ----
        elif "adj" in data:
            nx_g = nx.from_numpy_array(data["adj"])

        else:
            raise ValueError("Unknown graph format in pt file")

        nx_g = nx.Graph(nx_g)
        dataset.append(nx_g)
        
    node_counts = [g.number_of_nodes() for g in dataset]
    NVertices = min(node_counts)

    aligned_dataset = []

    for g in dataset:
        nodes_sorted = sorted(g.nodes())
        nodes_keep = nodes_sorted[:NVertices]
        g_sub = g.subgraph(nodes_keep).copy()
        # g_sub = nx.convert_node_labels_to_integers(g_sub)
        g_relabel = nx.convert_node_labels_to_integers(g_sub, ordering='sorted')
        aligned_dataset.append(g_relabel)

    dataset = aligned_dataset

elif data_name == 'DBLP_adj':
    data_dir = "data/DBLP_adj" 
    dataset = []

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith("_adj.npz")],
        key=lambda x: int(re.search(r'graph_(\d+)_adj\.npz', x).group(1))
        if re.search(r'graph_(\d+)_adj\.npz', x) else x
    )

    print("Found files:", files)

    for f in files:
        path = os.path.join(data_dir, f)
        mat = sparse.load_npz(path)
        nx_g = nx.from_scipy_sparse_array(mat)
        nx_g = nx_g.to_undirected()
        nx_g = nx.Graph(nx_g)

        dataset.append(nx_g)

    print(f"Loaded {len(dataset)} graphs from adj.npz files")

    node_counts = [g.number_of_nodes() for g in dataset]
    print("Node counts:", node_counts)

    NVertices = min(node_counts)
    print("Min number of nodes =", NVertices)

    aligned_dataset = []
    for g in dataset:
        nodes_sorted = sorted(g.nodes())
        nodes_keep = nodes_sorted[:NVertices]
        g_sub = g.subgraph(nodes_keep).copy()
        g_relabel = nx.convert_node_labels_to_integers(g_sub, ordering='sorted')
        aligned_dataset.append(g_relabel)

    dataset = aligned_dataset

for i, g in enumerate(dataset):
    print(i, g.number_of_nodes())


path = os.getcwd()

# NVertices = 307 # Number of vertices
scaleParameter = 1.0 # Scale Parameter (Maximum) # the maximal edge weight #
maxDimHoles = 2 # Maximum Dimension of Holes (It means.. 0 and 1)
sizeWindow = 5 # Number of Graphs

def edge_weight_function(G, alpha):
    deg = dict(G.degree())
    alpha = alpha

    scores = {(u, v): (deg[u] * deg[v]) ** alpha for u, v in G.edges()}
    mx = max(scores.values()) if scores else 1.0

    # 写入 edge weight
    for (u, v), s in scores.items():
        G[u][v]["weight"] = s / mx  # (0,1]

    return G

def zigzag_SBM_persistence_diagrams(dataset, NVertices, scaleParameter, maxDimHoles, sizeWindow, output_folder):
    #  Generate Graph
    # GraphsNetX = []
    # window_PD = []
    assert sizeWindow <= len(dataset), "window size is exceed the dataset length, can not process!"
    start_index = sizeWindow - 1
    while start_index < len(dataset):
        GraphsNetX = []
        print(f"######## compute the window from {start_index - sizeWindow + 1} to {start_index} ########")
        for ii in range(start_index - sizeWindow + 1, start_index):
            # print(f"compute the window from {start_index - sizeWindow + 1} to {start_index}")
            tmp_g = dataset[ii]
            target_g = nx.empty_graph(NVertices)
            for u, v, w in tmp_g.edges(data="weight"):
                target_g.add_edge(int(u), int(v), weight = w)

            GraphsNetX.append(target_g)

        #  Building unions and computing distance matrices
        GUnions = []
        MDisGUnions = []
        for i in range(0, sizeWindow - 2):
            # --- To concatenate graphs
            unionAux = []
            MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
            A = nx.adjacency_matrix(GraphsNetX[i]).todense()
            B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
            # ----- Version Original (2)
            C = (A + B) / 2
            A[A == 0] = 1.1
            A[range(NVertices), range(NVertices)] = 0
            B[B == 0] = 1.1
            B[range(NVertices), range(NVertices)] = 0
            MDisAux[0:NVertices, 0:NVertices] = A
            C[C == 0] = 1.1
            C[range(NVertices), range(NVertices)] = 0
            MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
            MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
            MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()
            # Distance in condensed form
            pDisAux = squareform(MDisAux)
            # --- To save unions and distances
            GUnions.append(unionAux)  # To save union
            MDisGUnions.append(pDisAux)  # To save distance matrix

        #  To perform Ripser computations
        GVRips = []
        for jj in range(0, sizeWindow - 2):
            print(jj)
            ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter)
            GVRips.append(ripsAux)

        #  Shifting filtrations...
        GVRips_shift = []
        GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
        for kk in range(1, sizeWindow - 2):
            shiftAux = zzt.shift_filtration(GVRips[kk], NVertices * kk)
            GVRips_shift.append(shiftAux)
        print(f"shift array size is {len(GVRips_shift)}")


        completeGVRips = GVRips_shift[0]
        for uu in range(1, len(GVRips_shift)):
            completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
            
        #  To Combine complexes
        # completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
        # if sizeWindow >= 4:
        #     for uu in range(2, sizeWindow - 2):
        #         completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])

        #  To compute the time intervals of simplices
        time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)

        #  To compute Zigzag persistence
        G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
        print("  --- End Zigzag")  # Beginning

        #  To show persistence intervals
        window_PD = []

        #  Personalized plot
        for vv, dgm in enumerate(G_dgms):
            if (vv < 2):
                matBarcode = np.zeros((len(dgm), 2))
                k = 0
                for p in dgm:
                    matBarcode[k, 0] = p.birth
                    matBarcode[k, 1] = p.death
                    k = k + 1
                matBarcode = matBarcode / 2  ## final PD!!! ##
                window_PD.append(matBarcode)  # return list form
        np.save(f"{output_folder}/{start_index - sizeWindow + 1}_{start_index}_pd.npy", np.array(window_PD, dtype=object))
        start_index = start_index + 1
    return window_PD

alpha = 0.5
edge_weight_dataset = []
for graph in dataset:
    edge_weight_dataset.append(edge_weight_function(graph, alpha))
output_folder = f"output/{data_name}_m{graph_class}/graph_{graph_index}/"
os.makedirs(output_folder, exist_ok=True)

# test = zigzag_SBM_persistence_diagrams(dataset = [edge_weight_function(g_0, alpha), edge_weight_function(g_1, alpha),
#                                                   edge_weight_function(g_2, alpha), edge_weight_function(g_3, alpha),
#                                                   edge_weight_function(g_4, alpha)], NVertices=300, scaleParameter=1.0, maxDimHoles=2, sizeWindow = 5)

test = zigzag_SBM_persistence_diagrams(dataset = edge_weight_dataset, NVertices=NVertices, scaleParameter=1.0, maxDimHoles=2, sizeWindow = sizeWindow, output_folder = output_folder)
