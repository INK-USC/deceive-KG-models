import numpy as np
import os
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import networkx as nx
import pickle
from val_score_reco import get_dist
import torch
from predict import *
import collections
import os
import numpy as np

def load_data(args, type='rel', num_pert=0):
    n_user, n_item, train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args, type, num_pert)
    #print(n_entity)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    print('data loaded.')

    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data, user_history_dict = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data, user_history_dict


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict



def load_kg(args, type = 'rel', num_pert=0):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_initial'
    if os.path.exists(kg_file + '.npy'):
        print('Found npy')
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    print(n_entity, n_relation)

    kg = construct_kg(kg_np, n_entity, n_relation, type, num_pert)
    get_dist(kg, n_entity, n_relation)
    #adj_entity, adj_relation = construct_adj(args, kg, n_entity, type)

    return n_entity, n_relation, kg


def construct_kg(kg_np, n_ent, n_rel,type = 'rel', num_pert=0):
    print('constructing knowledge graph ...')
    print("Total_edges_before: ", kg_np.shape[0])
    print('making ',num_pert,' perturbations')
    kg = dict()
    seen = []

    perturbations_edge = random.sample(range(kg_np.shape[0]), num_pert)
    perts = {}
    for i in perturbations_edge:
        perts[i] = 1
    i=-1
    for triple in tqdm(kg_np):
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if type=='edge1' or type=='no_kg':
            i = i+1
            try:
                if perts[i]==1:
                    if head not in kg:
                        kg[head] = []
                    kg[head].append((0,0))
                    continue
            except KeyError:
                if head not in kg:
                    kg[head] = []
                kg[head].append((tail, relation))
                # if tail not in kg:
                #     kg[tail] = []
                # kg[tail].append((head, relation))
        elif type=='rel1':
            i = i+1
            try:
                if perts[i]==1:
                    relation = get_relation(head, tail, n_rel)
            except KeyError:
                relation = relation
            if head not in kg:
                kg[head] = []
                #relation = get_relation(head, tail, n_rel)
            kg[head].append((tail, relation))
            # if tail not in kg:
            #     kg[tail] = []
            # kg[tail].append((head, relation))
        else:
            if head not in kg:
                kg[head] = []
            kg[head].append((tail, relation))
            # if tail not in kg:
            #     kg[tail] = []
            # kg[tail].append((head, relation))

    if type == 'edge2':
        for i in tqdm(range(num_pert)):
            node_1 = random.choice(list(kg.keys()))
            while len(kg[node_1])==0:
                node_1 = random.choice(list(kg.keys()))
            tuple = random.choice(range(len(kg[node_1])))
            edge = kg[node_1][tuple]
            kg[node_1].remove(kg[node_1][tuple])
            # kg[edge[0]].remove((node_1, edge[1]))
            ngbrs = [ent for _, ent in kg[node_1]]
            node_2 = random.choice(list(kg.keys()))
            # while node_2 in ngbrs or node_2 == node_1:
            #     node_2 = random.choice(range(n_ent))
            kg[node_1].append((node_2, edge[1]))
            # kg[node_2].append((node_1, edge[1]))

    if type == 'rel':
        for i in tqdm(range(num_pert//2)):
            perturbations = random.sample(range(kg_np.shape[0]), 2)
            while perturbations in seen:
                perturbations = random.sample(range(kg_np.shape[0]), 2)
            seen.append(perturbations)
            adj_1 = kg[kg_np[perturbations[0]][0]]
            for j in range(len(adj_1)):
                if adj_1[j][1] == kg_np[perturbations[0]][1] and adj_1[j][0] == kg_np[perturbations[0]][2]:
                    kg[kg_np[perturbations[0]][0]][j] =  (kg_np[perturbations[0]][2], kg_np[perturbations[1]][1])
                    #kg[kg_np[perturbations[0]]['head']][j][1] =  kg_np[perturbations[1]]['tail']
            adj_2 = kg[kg_np[perturbations[0]][2]]
            for j in range(len(adj_2)):
                if adj_2[j][1] == kg_np[perturbations[0]][1] and adj_2[j][0] == kg_np[perturbations[0]][0]:
                    kg[kg_np[perturbations[0]][2]][j] =  (kg_np[perturbations[0]][0], kg_np[perturbations[1]][1])
                    #kg[kg_np[perturbations[0]]['tail']][j][1] =  kg_np[perturbations[1]]['head']

            adj_1 = kg[kg_np[perturbations[1]][0]]
            for j in range(len(adj_1)):
                if adj_1[j][1] == kg_np[perturbations[1]][1] and adj_1[j][0] == kg_np[perturbations[1]][2]:
                    kg[kg_np[perturbations[1]][0]][j] =  (kg_np[perturbations[1]][2], kg_np[perturbations[0]][1])
                    #kg[kg_np[perturbations[1]]['head']][j][1] =  kg_np[perturbations[0]]['tail']
            adj_2 = kg[kg_np[perturbations[1]][2]]
            for j in range(len(adj_2)):
                if adj_2[j][1] == kg_np[perturbations[1]][1] and adj_2[j][0] == kg_np[perturbations[0]][0]:
                    kg[kg_np[perturbations[1]][2]][j] =  (kg_np[perturbations[1]][0], kg_np[perturbations[0]][1])
    valid_score(kg)
    return kg


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                try:
                    tails_of_last_hop = ripple_set[user][-1][2]
                except IndexError:
                    tails_of_last_hop = user_history_dict[user]

            for entity in tails_of_last_hop:
                try:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])
                except KeyError:
                    continue

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                try:
                    ripple_set[user].append(ripple_set[user][-1])
                except:
                    continue
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set

def valid_score(graph):
    #data = np.zeros((nrow_positive,4))
    c = 0
    # with open(kg_path, 'rb') as handle:
    #     graph = pickle.load(handle)
    for cpt in list(graph.keys()):
        c = c + len(graph[cpt])
    data = np.zeros((c,4))
    i = 0
    yu = 0
    for cpt in tqdm(list(graph.keys())):
        for triple in graph[cpt]:
            yu = yu + pred(cpt, triple[1], triple[0])

            #i = i+1
    #model1 = load_model("deep_classifier_1.hdf5")
    # print(data[:i].shape)
    # scores = model1.predict(data[:i,:3])
    # print(scores.shape)
    print("validation_score: ",yu/c)

def get_relation(head, tail, n_rel):
    least_score = 1.2
    k=0
    for i in range(n_rel):
        val_score = pred(head, i, tail)
        if val_score.item()<least_score:
            least_score = val_score.item()
            k = i
    return i
