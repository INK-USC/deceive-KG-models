import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import random
from create_new_vocab import *

global concept2id, id2concept, vocab_index
with open('./data/cpnet/concept.txt', "r", encoding="utf8") as fin:
    id2concept = [w.strip() for w in fin]
concept2id = {w: i for i, w in enumerate(id2concept)}

my_file=open('new_vocab_2.txt', "r")
content = my_file.read()
vocab_index = content.split("\n")
while vocab_index[-1]=='':
    vocab_index=vocab_index[:-1]
vocab_index=[int(x) for x in vocab_index]

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

def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_path, prob = 0, prune=False):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    # concept2id = {}
    # id2concept = {}
    # with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
    #     id2concept = [w.strip() for w in fin]
    # concept2id = {w: i for i, w in enumerate(id2concept)}
    #
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    # del_cpts = random.sample(range(780000), prob)
    # del_cpts_dict = np.zeros((800000,))
    # del_cpts_dict[del_cpts] = 1
    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            # if cpt in blacklist or del_cpts_dict[concept2id[cpt]] == 1:
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()
        i = 0
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj not in vocab_index or obj not in vocab_index:
                continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???
            # if prune and i<num_changes:
            #     p = random.random()
            #     if p<0.5:
            #         rel = random.choice(list(range(len(relation2id))))
            #         i = i+1

            if (subj, obj, rel) not in attrs:
                # p = random.random()
                # if p<prob:
                #     i = i+1
                #     continue
                #     rel = random.choice(list(range(len(relation2id))))

                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=(rel + len(relation2id)), weight=weight)
                attrs.add((obj, subj, (rel + len(relation2id))))
    # print(i, " perturbations done")
    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()

def main():
    extract_vocab('./data/obqa/grounded/train.grounded.jsonl', './data/obqa/grounded/dev.grounded.jsonl', './data/obqa/grounded/test.grounded.jsonl', 'new_vocab_2.txt')
    construct_graph('./data/cpnet/conceptnet.en.csv', './data/cpnet/concept.txt', './data/cpnet/conceptnet_2.en.pruned.graph')

if __name__ == '__main__':
    main()
