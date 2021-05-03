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

import torch
import numpy as np
import time

def compute_kernel(x,y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x_tile = x.view(x_size,1,dim)
    x_tile = x_tile.repeat(1,y_size,1)
    y_tile = y.view(1,y_size,dim)
    y_tile = y_tile.repeat(x_size,1,1)
    return torch.exp(-torch.mean((x_tile-y_tile)**2,dim = 2)/float(dim))


def compute_mmd(x,y):
    x_kernel = compute_kernel(x,x)
    # print(x_kernel)
    y_kernel = compute_kernel(y,y)
    # print(y_kernel)
    xy_kernel = compute_kernel(x,y)
    # print(xy_kernel)
    return torch.mean(x_kernel)+torch.mean(y_kernel)-2*torch.mean(xy_kernel)


def load_merge_relation():
    relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def load_resources(cpnet_vocab_path):
    merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
    ]
    concept2id = {}
    relation2id = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    return concept2id, relation2id, id2concept, id2relation

def get_cpnet_simple(nx_graph):
    cpnet_simple = nx.Graph()
    for u, v, data in nx_graph.edges(data=True):
        if data['rel']>=17:
            continue
        w = data['rel']%17
        if cpnet_simple.has_edge(u, v):
            continue
        else:
            cpnet_simple.add_edge(u, v, weight=1)
    return cpnet_simple

def get_dist():
    ori_cc = np.load('ori_stats.npy')
    with open('./data/cpnet/conceptnet.en.pruned.graph', 'rb') as handle:
        graph = pickle.load(handle)
    concept2id, relation2id, id2concept, id2relation = load_resources('./data/cpnet/concept.txt')
    coeff = np.zeros((ori_cc.shape[0],))
    cpnet_simple = get_cpnet_simple(graph)
    cc = nx.clustering(cpnet_simple)
    #v = torch.FloatTensor(list(cc.values()))
    #print(v.shape, ori_stats.shape)
    #print(np.max(list(cc.keys())))
    #print(np.max(list(graph.nodes)))
    for node, val in cc.items():
        coeff[node] = val
    #ori_stats = ori_stats.numpy()
    #np.save('ori_stats.npy', coeff)
    print(np.sum((ori_cc - coeff)**2))
    #print(compute_mmd(ori_stats.view(ori_stats.shape[0],1),v.view(v.shape[0],1)))
    #torch.save(v, 'ori_graph_cc.pt')
    # edges = list(graph.edges.data())
    # for rel_id in id2relation:
    #     cpnet_simple = nx.Graph()
    #     i = 0
    #     for edge in tqdm(edges):
    #         try:
    #             rel = edge[2]['rel']
    #             #print(rel)
    #             if rel>=17:
    #                 continue
    #             if rel==relation2id[rel_id]:
    #                 subj = edge[0]
    #                 obj = edge[1]
    #                 #print('k')
    #                 w = 1.0
    #                 #if not cpnet_simple.has_edge(subj, obj):
    #                 cpnet_simple.add_edge(subj, obj, weight=w)
    #                 i=i+1
    #         except KeyError:
    #             continue
    #     cc = nx.clustering(cpnet_simple)
    #     coeff.append(torch.FloatTensor(list(cc.values())))
    #     print(i)
    # new_stats = []
    # for i in range(17):
    #     new_stats.append(compute_mmd(ori_stats[i].view(ori_stats[i].shape[0],1),coeff[i].view(coeff[i].shape[0],1)))
    # print(np.mean(new_stats))

if __name__ == '__main__':
    get_dist()
