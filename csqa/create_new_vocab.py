import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np

global concept2id, id2concept, cpnet
cpnet = nx.read_gpickle('./data/cpnet/conceptnet.en.pruned.graph')
# cpnet_simple = get_cpnet_simple(cpnet)
with open('./data/cpnet/concept.txt', "r", encoding="utf8") as fin:
    id2concept = [w.strip() for w in fin]
concept2id = {w: i for i, w in enumerate(id2concept)}

def extract_vocab(grounded_path_1, grounded_path_2, grounded_path_3, output_path):
    cpts = set()
    entities = list(cpnet.nodes)
    with open(grounded_path_1, 'r') as fin:
        data = [json.loads(line) for line in fin]
    for line in tqdm(data):
        for source in line['qc']:
            if concept2id[source] in entities:
                cpts.add(concept2id[source])
        for target in line['ac']:
            if concept2id[target] in entities:
                cpts.add(concept2id[target])

    with open(grounded_path_2, 'r') as fin:
        data = [json.loads(line) for line in fin]
    for line in tqdm(data):
        for source in line['qc']:
            if concept2id[source] in entities:
                cpts.add(concept2id[source])
        for target in line['ac']:
            if concept2id[target] in entities:
                cpts.add(concept2id[target])

    with open(grounded_path_3, 'r') as fin:
        data = [json.loads(line) for line in fin]
    for line in tqdm(data):
        for source in line['qc']:
            if concept2id[source] in entities:
                cpts.add(concept2id[source])
        for target in line['ac']:
            if concept2id[target] in entities:
                cpts.add(concept2id[target])
    with open(output_path, 'w') as f:
        for item in list(cpts):
            f.write("%s\n" % item)
    print(len(list(cpts)))

def main():
    extract_vocab('./data/csqa/grounded/train.grounded.jsonl', './data/csqa/grounded/dev.grounded.jsonl', './data/csqa/grounded/test.grounded.jsonl', 'new_vocab_1.txt')

if __name__ == '__main__':
    main()
