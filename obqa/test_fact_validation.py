from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from multiprocessing import cpu_count
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
import tensorflow as tf
import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout, Input, concatenate
from keras.optimizers import Adam, SGD, Adagrad
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models import Sequential, load_model
from tqdm import tqdm
import pickle

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 20} )
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)




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
    return concept2id, relation2id, id2concept

def main_fn():
    concept2id, relation2id, id2concept = load_resources('./data/cpnet/concept.txt')
    ent_emb = np.load('./data/cpnet/tzw.ent.npy')
    rel_emb = np.load('./data/transe/glove.transe.sgd.ent.npy')
    nrow_positive = sum(1 for _ in open('./data/cpnet/conceptnet.en.csv', 'r', encoding='utf-8'))
    #nrow_negative = sum(1 for _ in open('neg_triples.csv', 'r', encoding='utf-8'))
    data = np.zeros((nrow_positive,4))
    i = 0
    rel_mapper = load_merge_relation()
    with open('./data/cpnet/conceptnet.en.pruned.graph', 'rb') as handle:
        graph = pickle.load(handle)
    edges = list(graph.edges.data())
    for edge in tqdm(edges):
        #ls = line.strip().split('\t')
        try:
            rel = edge[2]['rel']
            if rel>=17:
                continue
            subj = edge[0]
            obj = edge[1]
            #label = int(ls[3])
            #if concept2id[subj] == 0:
            #    data[i][0] = 0
            #else:
            #    data[i][0] = concept2id[subj]
            #if relation2id[rel] == 6:
            #    data[i][1] = relation2id[rel] + len(concept2id)
            #else:
            #    data[i][1] = relation2id[rel] + len(concept2id)
            #data[i][2] = concept2id[obj]
            #data[i][3] = 0
            if rel == 0:
                data[i][0] = subj
                data[i][1] = 0 + len(concept2id)
                data[i][2] = obj
            elif rel == 15:
                data[i][0] = subj
                data[i][1] = 15 + len(concept2id)
                data[i][2] = obj
            else:
                data[i][0] = subj
                data[i][1] = rel + len(concept2id)
                data[i][2] = obj
            data[1][3] = 0
            i = i + 1
        except KeyError:
            continue

    Y_test = data[:i,3]
    model1 = load_model("deep_classifier_1.hdf5")
    print(data[:i].shape)
    scores = model1.predict(data[:i,:3])
    print(scores.shape)
    print(np.mean(scores))
    # loss, accuracy = model1.evaluate(data[:i,:3], Y_test)
    x = np.zeros((1,3))
    x[0,0] = 30848
    x[0,1] = len(concept2id)
    x[0,2] = 80217
    #print(i)
    print(id2concept[30848])
    print(id2concept[80217])
    list1 = []
    for rel in list(relation2id.keys()):
        x[0,1] = len(concept2id) + relation2id[rel]
        list1.append((rel,model1.predict(x)[0]))
    print(list1)

if __name__ == '__main__':
    main_fn()
