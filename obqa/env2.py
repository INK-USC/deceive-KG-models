import os
import sys
import numpy as np
from Replay_Memory import *
import tensorflow as tf
from newprocess import *
import networkx as nx
from rn import *
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
from multiprocessing import Pool
import copy

class NodeAttakEnv(object):
    def __init__(self, vocab, relations, max_steps, init_score, init_acc, update_step, init_graph, init_avg_prob):
        self.vocab = vocab
        self.relations = relations
        self.n_steps = 0
        self.max_steps = max_steps
        self.update_step = update_step
        self.triple_classifier = load_model("deep_classifier_1.hdf5")
        self.score = init_score
        self.prev_score = init_score
        self.init_avg_prob = init_avg_prob
        self.prev_avg_prob = init_avg_prob
        self.init_score = init_score
        self.update_step_score = init_score

        self.graph = copy.deepcopy(init_graph)
        self.num_edges = self.graph.number_of_edges()
        #self.action = []
        #self.subjs = None
        #self.objs = None
        #self.modified_list = []

    def is_Terminal(self):
        if self.n_steps == self.update_step:
            return 1
        return 0

    def reset(self):
        self.n_steps = 0
        self.score = self.init_score
        self.prev_score = self.init_score
        self.num_edges = self.graph.number_of_edges()
        self.update_step_score = self.init_score
        self.prev_avg_prob = self.init_avg_prob
        with open('./data/cpnet/conceptnet_2.en.pruned.graph', 'rb') as handle:
            self.graph = pickle.load(handle)

    def perturb_relation(self, edge):
        source = int(edge[0])
        target = int(edge[1])
        rel = int(edge[2]['rel'])
        #rel1 = rel
        # try:
        rel_list = copy.deepcopy(self.graph[source][target])
        # if rel_list[0]['rel'] < 17:
        #print(rel_list)
        rel1 = copy.deepcopy(rel_list[0]['rel'])
        #print(edge[2]['rel'], rel1)
        self.graph[int(source)][int(target)][0]['rel'] = int(rel)
        self.graph[int(target)][int(source)][0]['rel'] = int(17+rel)
        # else:
        #     rel1 = rel_list[1]['rel']
        #     self.graph[source][target][1]['rel'] = rel
        #     self.graph[target][source][0]['rel'] = 17+rel
        return rel1
        # except KeyError:
        #     print('Invalid Action')
        #     return None



    def step(self,action, state, is_train = True, model = None):
        # action is subj,obj,rel
        x = np.zeros((1,3))
        x[0,0] = action[0]
        x[0,1] = len(self.vocab) + action[2]['rel']
        x[0,2] = action[1]
        # score of the action to be taken
        #print(x)
        act_score = self.triple_classifier.predict(x)[0]
        #act_score = 0
        #print(act_score)
        # total score of all triples before taking current action
        #num_edges = self.num_edges
        #len_state = len(state[1])
        #graph_score = (num_edges)*self.score

        #print('score')
        # if time to update the graph
        rel_1 = self.perturb_relation(action)
        if rel_1 is None:
            return state, 0, self.is_Terminal()
        if rel_1 < 17:
            x[0,0] = action[0]
            x[0,1] = len(self.vocab) + rel_1
            x[0,2] = action[1]
        else:
            x[0,0] = action[1]
            x[0,1] = len(self.vocab) + rel_1-17
            x[0,2] = action[0]
        changed_act_score = self.triple_classifier.predict(x)[0]
        # print((act_score - changed_act_score))
        self.score = ((self.prev_score *self.num_edges) + (act_score - changed_act_score))/self.num_edges
        #print(self.score)

        if self.n_steps%self.update_step == 0 and self.n_steps > 0:
            output_path = "./data/cpnet/conceptnet.en.pruned.graph"
            state[1].append(action)
            #self.graph = state[0]
            state[0] = self.graph
            #self.prev_score = self.score
            #self.score = (graph_score + act_score)/(num_edges + 1)   # update the average validation score
            nx.write_gpickle(self.graph, output_path)
            if is_train:
                main1()             # start to run the model
                # implement the below line
                prob_list = main2(mode = 'eval', model=model)
                # acc=self.init_acc
                prob_list = np.asarray(prob_list, dtype=np.float32)
                num_probs = np.sum(prob_list<0.2)
                prob_list = prob_list*(prob_list<0.2)

                avg_prob = np.sum(prob_list)/num_probs
                reward = 250* (avg_prob - self.prev_avg_prob) + 500*(self.update_step_score - self.score)[0]
                state[1] = []
                self.update_step_score = self.score
                self.prev_avg_prob = avg_prob
            else:
                reward = 0
        else:
            #reward = self.prev_score - self.score
            reward = 0
            state[1].append(action)
            #print('score')
            #self.prev_score = self.score
            #self.score = (graph_score + act_score)/(num_edges + 1)

        self.n_steps = self.n_steps + 1
        self.prev_score = self.score
        #self.num_edges = self.num_edges + 1
        return state, reward, self.is_Terminal(), self.score
