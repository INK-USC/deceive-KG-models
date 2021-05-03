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
import torch
import math
import random
from newprocess_2_RL import *
from gn_RL import *
from grn_RL import *
from newprocess_3 import *


def get_random_action0(graph, size, node_list):
    action0=random.randint(0,size-1)
    while len(graph.adj[node_list[action0]])==0:
        action0=random.randint(0,size-1)
    return node_list[action0]

def random_get_state():
    s1=random.getstate()
    s2=np.random.get_state()
    s3=torch.get_rng_state()
    s4=torch.cuda.get_rng_state_all()
    return (s1,s2,s3,s4)

def random_set_state(seed_state):
    s1,s2,s3,s4=seed_state
    random.setstate(s1)
    np.random.set_state(s2)
    torch.set_rng_state(s3)
    torch.cuda.set_rng_state_all(s4)


def get_prob_cost(prob_list):
    prob_cost=0
    for li in prob_list:
        co=li[5]
        co=int(co)
        # flag=1
        # for i in range(5):
        #     if li[co]<li[i]:
        #         flag=0
        # if flag:  #correct
        #     acc=acc+1
        for i in range(4):
            if i==co:
                prob_cost+=li[i]*math.log(li[i])
    #prob_cost=prob_cost/prob_list.shape[0]
    # acc=acc/prob_list.shape[0]
    return -prob_cost







class NodeAttakEnv(object):
    def __init__(self, args,vocab_index, vocab, relations, max_steps, init_score, init_acc, update_step, init_graph, init_avg_prob,hc0, device):
        self.args = args
        self.vocab = vocab
        self.vocab_index=np.array(vocab_index)
        self.relations = relations
        self.n_steps = 0
        self.max_steps = max_steps
        self.update_step = update_step
        self.triple_classifier = load_model("deep_classifier_1.hdf5")
        self.score = init_score
        self.prev_score = init_score
        self.init_avg_prob = init_avg_prob
        self.prev_avg_prob = init_avg_prob
        self.prev_acc = init_acc
        self.init_score = init_score
        self.update_step_score = init_score
        self.hc0=hc0
        self.device = device
        self.eval= (args.mode_type=='eval')

        self.graph = copy.deepcopy(init_graph)
        self.num_edges = self.graph.number_of_edges()
        self.graph2=copy.deepcopy(init_graph)
        print('number of edges in graph=', self.num_edges)


    def is_Terminal(self):
        if self.n_steps % self.update_step==0:
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

    def evaluate(self):
        self.eval=True

    def perturb_relation(self, edge):
        source = int(edge[0])
        target = int(edge[1])
        rel_list = copy.deepcopy(self.graph[source][target])
        idx=len(self.graph[source][target])-1
        rel1 = copy.deepcopy(rel_list[idx]['rel'])

        if self.args.model_id in [4,5,6,7]: #delete edge and add
            target2= int(edge[2]['rel'])

            self.graph.add_edge(source, target2, rel=rel1,weight=1.0)
            self.graph.add_edge(source, target2, rel=(rel1+17)%34,weight=1.0)
            self.graph.remove_edge(source,target)
            if source in self.graph.neighbors(target):
                self.graph.remove_edge(target, source)
            self.graph2.add_edge(source, target2, rel=rel1,weight=1.0)
            self.graph2.add_edge(source, target2, rel=(rel1+17)%34,weight=1.0)

        else:
            rel = int(edge[2]['rel'])

            self.graph[int(source)][int(target)][0]['rel'] = int(rel)
            self.graph[int(target)][int(source)][0]['rel'] = int(17+rel)

        return rel1


    #input:(batch,), output:(batch,17)
    def predict_score(self,actions0,actions1):
        x = np.zeros((actions0.shape[0],3))
        x[:,0] = actions0
        x[:,2] = actions1
        score=[]
        for i in range(17):
            x[:,1] = len(self.vocab) + i
            score.append(self.triple_classifier.predict(x))
        score=np.concatenate(score,-1)
        return score

    def least_edge_val(self, node1,rel,k=1,sample_num=1000):
        x=np.zeros((sample_num,3))
        if rel>=17:
            rel=rel%17
            x[:,0]=random.sample(list(self.vocab_index),sample_num)
            x[:,1]=rel+len(self.vocab)
            x[:,2]=node1
        else:
            x[:,0]=node1
            x[:,1]=rel+len(self.vocab)
            x[:,2]=random.sample(list(self.vocab_index),sample_num)
        score=self.triple_classifier.predict(x)
        if self.args.model_id==6:
            node2=self.vocab_index[np.argmin(score)]
            return node2
        elif self.args.model_id==5:
            nodes2=self.vocab_index[np.argsort(score)[:100]]
            return nodes2[:,0]

    def step(self, action, state, hc, model=None):
        # action is subj,obj,rel

        # score of the action to be taken
        #print(x)
        rel_1 = self.perturb_relation(action)
        if self.args.model_id not in [4,5,6,7]:
            x = np.zeros((1,3))
            x[0,0] = action[0]
            x[0,1] = len(self.vocab) + action[2]['rel']
            x[0,2] = action[1]
            act_score = self.triple_classifier.predict(x)[0]

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

        state[0]=self.graph
        state[2]=state[2]+1
        state[3]=hc
        if (self.n_steps+1)%self.update_step == 0 and self.n_steps > 0:
            state[2].zero_()
            state[3]=self.hc0
            state[1].append(action)

            output_path = "./data/cpnet/conceptnet.en.pruned.graph"
            #self.graph = state[0]
            #state[0] = self.graph
            #self.prev_score = self.score
            #self.score = (graph_score + act_score)/(num_edges + 1)   # update the average validation score
            nx.write_gpickle(self.graph, output_path)
            if self.eval or self.args.debug_mode:
                avg_prob=self.prev_avg_prob+0.5
                acc=0
                prob_list=[]
                acc_list=[]
            else:

                seed_state=random_get_state()

                if self.args.graph_model==0:
                    main1()
                    prob_list,acc_list = main2(mode = 'eval', model=model)
                elif self.args.graph_model==1:
                    main3()
                    prob_list,acc_list = main4(mode = 'eval', model=model)
                elif self.args.graph_model==2:
                    main5()
                    prob_list,acc_list = main6(mode = 'eval', model=model)
                else: 
                    print('Undefined graph_model!',args.graph_model)
                    exit()
                random_set_state(seed_state)
                avg_prob=get_prob_cost(prob_list)
                acc=np.mean(acc_list)

            reward = (avg_prob - self.prev_avg_prob) #+ 1000*(self.update_step_score - self.score)[0]
            # print('reward0=',reward, avg_prob, self.prev_avg_prob)
            state[1] = []
            self.update_step_score = self.score
            self.prev_avg_prob = avg_prob
            self.prev_acc = acc
        else:
            #reward = self.prev_score - self.score

            reward = 0
            acc=0
            avg_prob=0
            state[1].append(action)
            prob_list=[]
            #print('score')
            #self.prev_score = self.score
            #self.score = (graph_score + act_score)/(num_edges + 1)

        self.n_steps = self.n_steps + 1
        self.prev_score = self.score
        #self.num_edges = self.num_edges + 1
        return state, reward, self.is_Terminal(), self.score, acc,  avg_prob, prob_list, self.graph2
