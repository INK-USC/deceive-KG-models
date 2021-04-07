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
    return concept2id, relation2id 

def main():
    concept2id, relation2id = load_resources('./data/cpnet/concept.txt')
    num_cpts = len(list(concept2id.keys()))
    vocab_size = len(list(concept2id.keys())) + 17
    emb_mat = np.zeros((vocab_size,100))
    ent_emb = np.load('./data/transe/glove.transe.sgd.ent.npy')
    rel_emb = np.load('./data/transe/glove.transe.sgd.rel.npy')
    emb_mat[:num_cpts] = ent_emb
    emb_mat[num_cpts:] = rel_emb
    nrow_positive = sum(1 for _ in open('./data/cpnet/conceptnet.en.csv', 'r', encoding='utf-8'))
    nrow_negative = sum(1 for _ in open('neg_triples.csv', 'r', encoding='utf-8'))
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=3, weights = [emb_mat],trainable=True))
    model.add(Flatten())
    model.add(Dense(1000,activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    data = np.zeros((nrow_negative + nrow_positive,4))
    i = 0;
    with open('./data/cpnet/conceptnet.en.csv', "r", encoding="utf8") as fin:
        for line in tqdm(fin, total=nrow_positive):
            ls = line.strip().split('\t')
            rel = ls[0]
            subj = ls[1]
            obj = ls[2]
            data[i][0] = concept2id[subj]
            data[i][1] = relation2id[rel] + num_cpts
            data[i][2] = concept2id[obj]
            data[i][3] = 1
            i= i+1
            
    with open('neg_triples.csv', "r", encoding="utf8") as fin:
        for line in tqdm(fin, total=nrow_negative):
            ls = line.strip().split('\t')
            rel = ls[1]
            subj = ls[0]
            obj = ls[2]
            data[i][0] = concept2id[subj]
            data[i][1] = relation2id[rel] + num_cpts
            data[i][2] = concept2id[obj]
            data[i][3] = 0
            i= i+1
            
    np.random.shuffle(data)
    print('dataset created')
    X = data[(nrow_negative + nrow_positive)//5:,:3]
    Y = data[(nrow_negative + nrow_positive)//5:,3]
    X_valid = data[:(nrow_negative + nrow_positive)//10,:3]
    Y_valid = data[:(nrow_negative + nrow_positive)//10,3]
    X_test = data[(nrow_negative + nrow_positive)//10:(nrow_negative + nrow_positive)//5,:3]
    Y_test = data[(nrow_negative + nrow_positive)//10:(nrow_negative + nrow_positive)//5,3]
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    print(model.summary())
    model.fit(X, Y, epochs=5, verbose=1, batch_size=500,validation_data = (X_valid, Y_valid) )
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(accuracy)
    model.save('deep_classifier_1.hdf5')
    
    
if __name__ == '__main__':
    main()