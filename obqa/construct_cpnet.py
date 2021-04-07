import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import random
import pickle

def perturb_cpnet(graph_path, num_pert, output_path):
    with open(graph_path, 'rb') as handle:
        graph = pickle.load(handle)
    forward_edges = []
    all_edges = list(graph.edges.data())
    for edge in all_edges:
        if edge[2]['rel'] < 17:
            forward_edges.append(edge)

    #perturbations = []
    num_edges = len(forward_edges)
    #perturbations = random.sample(range(len(forward_edges)), num_pert)
    for i in tqdm(range(num_pert//2)):
        perturbations = random.sample(forward_edges, 2)
        graph[perturbations[0][0]][perturbations[0][1]][0]['rel'] = perturbations[1][2]['rel']
        #graph[perturbations[0][0]][perturbations[0][1]][0]['rel'] = perturbations[1][2]['rel'] + 17
        graph[perturbations[1][0]][perturbations[1][1]][0]['rel'] = perturbations[0][2]['rel']
        #graph[perturbations[1][0]][perturbations[1][1]][0]['rel'] = perturbations[0][2]['rel'] + 17

    nx.write_gpickle(graph, output_path)

    #new_graph = nx.MultiDiGraph()
