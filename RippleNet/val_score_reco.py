from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from multiprocessing import cpu_count
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import networkx as nx
from scipy.stats import entropy
import torch
import numpy as np
import time

def get_dist(graph, n_entity, n_rel):
    ori_stats = np.load('rel_stats.npy')
    # with open('kg.pk', 'rb') as handle:
    #     graph = pickle.load(handle)
    coeff = np.zeros(ori_stats.shape)
    # coeff = np.zeros((n_rel, n_entity))
    for rel_id in range(n_rel):
        cpnet_simple = nx.Graph()
        for cpt in list(graph.keys()):
            for triple in graph[cpt]:
                if triple[1] == rel_id:
                    #print("k")
                    cpnet_simple.add_edge(cpt, triple[1], weight=1)
        cc = cpnet_simple.degree()
        # try:
        #     print(len(cpnet_simple[2241]))
        # except KeyError:
        #     print('f')
        for node, val in list(cc):
            coeff[rel_id, node] = val
    # print(np.where(coeff != ori_stats))
    #print(coeff[:,2241])
    cv = np.sqrt(np.sum((coeff - ori_stats)**2, axis=1))
    print(cv.shape)
    print(1/(np.mean(cv) + 0.00001))
    # np.save('rel_stats.npy', coeff)

if __name__ == '__main__':
    get_dist()
