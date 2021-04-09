import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import random
import pickle

def perturb_cpnet(graph_path, num_pert, output_path, type):
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
    seen = []
    if type=='rel':
        for i in tqdm(range(num_pert//2)):
            perturbations = random.sample(forward_edges, 2)
            while len(seen)>0 and perturbations in seen:
                perturbations = random.sample(forward_edges, 2)
            graph[perturbations[0][0]][perturbations[0][1]][0]['rel'] = perturbations[1][2]['rel']
            #graph[perturbations[0][0]][perturbations[0][1]][0]['rel'] = perturbations[1][2]['rel'] + 17
            graph[perturbations[1][0]][perturbations[1][1]][0]['rel'] = perturbations[0][2]['rel']
            seen.append(perturbations)
        #graph[perturbations[1][0]][perturbations[1][1]][0]['rel'] = perturbations[0][2]['rel'] + 17
    elif type=='edge':
        perturbations = random.sample(range(len(forward_edges)), num_pert)
        dict = {}
        for edge in perturbations:
            dict[edge] = 1
        graph = nx.MultiDiGraph()
        for edge in tqdm(range(len(forward_edges))):
            try:
                if dict[edge]==1:
                    continue
            except KeyError:
                graph.add_edge(forward_edges[edge][0], forward_edges[edge][1], rel=forward_edges[edge][2]['rel'], weight=forward_edges[edge][2]['weight'])
                graph.add_edge(forward_edges[edge][1], forward_edges[edge][0], rel=forward_edges[edge][2]['rel']+17, weight=forward_edges[edge][2]['weight'])

    elif type=='edge1':
        nodes = set()
        for edge in forward_edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        for i in tqdm(range(num_pert)):
            node_1 = random.choice(tuple(nodes))
            ngbrs = set(list(graph.neighbors(node_1)))
            while len(ngbrs)==0:
                node_1 = random.choice(tuple(nodes))
                ngbrs = set(list(graph.neighbors(node_1)))
            node_2 = random.choice(tuple(ngbrs))
            non_ngbrs = nodes - ngbrs
            node_3 = random.choice(tuple(non_ngbrs))
            edge = graph[node_1][node_2][0]
            #print(edge)
            graph.remove_edge(node_1, node_2)
            rev_edge = (node_2, node_1)
            if node_1 in graph.neighbors(node_2):
                graph.remove_edge(node_2, node_1)
            graph.add_edge(node_1, node_3, rel= edge['rel'], weight=edge['weight'])
            graph.add_edge(node_3, node_1, rel= (edge['rel']+17)%34, weight=edge['weight'])
        # all_edges = list(graph.edges.data())
        # graph_2 = nx.MultiDiGraph()
        # for edge in all_edges:
        #     graph_2.add_edge(edge[0], edge[1], rel= edge[2]['rel'], weight=edge[2]['weight'])
        # graph = graph_2
    nx.write_gpickle(graph, output_path)

    #new_graph = nx.MultiDiGraph()
