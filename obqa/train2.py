
import random
from multiprocessing import cpu_count
from utils.conceptnet import extract_english, construct_graph
#from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)

from modeling.modeling_rn import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils.relpath_utils import *

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle

from dqn import *
from env import *
from newprocess import *
from deep_triple_classifier import *
from Replay_Memory import *
import argparse
import time
import torch
from rn import *
from test_script import *

# 60.5 without KG

class TimeTickTock():
    def __init__(self):
        self.reset()
        self._enable=True
        self.setting()

    def tick(self):
        self.tic = time.time()

    def tock(self, message=None):
        if self._enable:
            self.messages.append(message)
            self.times.append(time.time()-self.tic)
        self.tic = time.time()

    def reset(self):
        self.times=[]
        self.messages=[]
        self.tic = time.time()

    def show(self, demical=None, show_messages=None, separate=None, show_total_time=True):
        if not self._enable:
            return
        print(locals())
        #self.updata_settings(locals())
        if separate is None:
            separate = self._separate
        if show_messages is None:
            show_messages = self._show_messages
        if demical is None:
            demical = self._demical

        cout_string=''
        if show_messages:
            for  ti,mess in zip(self.times,self.messages):
                if mess is None:
                    cout_string+=str(round(ti,demical))+ separate
                else:
                    cout_string+=mess+':'+ str(round(ti,demical)) + separate

        else:
            for ti in self.times:
                cout_string+=str(round(ti,demical))+separate
        if show_total_time:
            cout_string+='total_time='+str(round(sum(self.times),demical))
        print(cout_string)

    def enable(self, enable=True):
        self._enable=enable

    def setting(self, separate=', ',demical=3,show_messages=True, show_total_time=True):
        self._separate = separate
        self._demical = demical
        self._show_messages = show_messages
        self._show_total_time=show_total_time




def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode_type', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--enable_TimeTickTock', action='store_true',default=False)
    parser.add_argument('--save_dir', default=f'./saved_models/KG/', help='model output directory')
    #parser.add_argument('--load_path', default=f'./saved_models/rn/', help='model output directory')

    parser.add_argument('-pertub_rate', '--pertub_rate', default=0.5, type=float, help='rate of pertubation')
    parser.add_argument('-update_rate', '--update_rate', default = 10, type = int)
    parser.add_argument('-epsilon_start', '--epsilon_start', default=1.0, type=float, help='epsilon_start')
    parser.add_argument('-epsilon_end', '--epsilon_end', default=0.01, type=float, help='epsilon_end')
    parser.add_argument('-epsilon_decay_steps', '--epsilon_decay_steps', default=5000, type=int, help='epsilon_decay_steps')
    parser.add_argument('-num_epochs', '--num_epochs', default = 20, type = int)
    parser.add_argument('-save_model_step', '--save_model_step', default = 1000, type = int)

    #  paths
    parser.add_argument('--classifier_path', default='./deep_classifier_1.hdf5')
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
    parser.add_argument('--cpnet_index', default='./new_vocab.txt')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    # data
    parser.add_argument('--train_rel_paths', default=f'./data/{args.dataset}/paths/train.relpath.2hop.jsonl')
    parser.add_argument('--dev_rel_paths', default=f'./data/{args.dataset}/paths/dev.relpath.2hop.jsonl')
    parser.add_argument('--test_rel_paths', default=f'./data/{args.dataset}/paths/test.relpath.2hop.jsonl')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
    # parser.add_argument('--train_node_features', default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    # parser.add_argument('--dev_node_features', default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    # parser.add_argument('--test_node_features', default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--cpt_emb', default='./data/transe/glove.transe.sgd.ent.npy')
    parser.add_argument('--relation_emb', default='./data/transe/glove.transe.sgd.rel.npy')
    parser.add_argument('--train_concepts', default=f'./data/{args.dataset}/grounded/train.grounded.jsonl')
    parser.add_argument('--dev_concepts', default=f'./data/{args.dataset}/grounded/dev.grounded.jsonl')
    parser.add_argument('--test_concepts', default=f'./data/{args.dataset}/grounded/test.grounded.jsonl')

    # optimization
    parser.add_argument('-lr', '--lr', default=3e-4, type=float, help='learning rate')
    #parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    #parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    args = parser.parse_args()
    train(args)


def train(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #construct_graph('./data/cpnet/conceptnet.en.csv', './data/cpnet/concept.txt', args.cpnet_graph_path, 0, True)
    #### load dataset: graph(networkx), vocab and relations and pre-embeddings, not implemented yet.
    with open('./data/cpnet/conceptnet_2.en.pruned.graph', 'rb') as handle:
        graph = pickle.load(handle)

    my_file = open(args.cpnet_vocab_path, "r")
    content = my_file.read()
    vocab = content.split("\n")
    vocab=vocab[:-1]
    my_file.close()

    my_file=open(args.cpnet_index, "r")
    content = my_file.read()
    vocab_index = content.split("\n")
    while vocab_index[-1]=='':
        vocab_index=vocab_index[:-1]
    vocab_index=[int(x) for x in vocab_index]
    #print('new vocab_size=', len(vocab_index))
    num_vocab_index=len(vocab_index)
    my_file.close()
    log_path = 'log.csv'
    with open(log_path, 'w') as fout:
        fout.write('step,reward,loss,score\n')

    relations = [
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

    nx.write_gpickle(graph, './data/cpnet/conceptnet.en.pruned.graph')
    #if args.mode_type=='train':
    main1()
    model = load_model()
    init_prob_list = main2(mode = 'eval', model = model)
    init_prob_list = np.asarray(init_prob_list, dtype=np.float32)
    num_probs = np.sum(init_prob_list<0.2)
    init_prob_list = init_prob_list*(init_prob_list<0.2)
    
    init_avg_prob = np.sum(init_prob_list)/num_probs
    #init_avg_prob = np.mean(init_prob_list)


    num_nodes = len(vocab)
    num_relations = len(relations)
    print('num_nodes=',num_nodes, ', num_vocab_index=',num_vocab_index, ', num_relations=', num_relations)

    node_pre_embedding = np.load(args.cpt_emb)
    relation_pre_embedding = np.load(args.relation_emb)

    print('node_pre_embedding=',node_pre_embedding.shape, ', relation_pre_embedding=', relation_pre_embedding.shape)
    ####
    #delayed_reward_step=(num_vocab_index * args.pertub_rate)//args.update_rate


    delayed_reward_step= 500
    print('delayed reward step=', delayed_reward_step)
    device=torch.device('cuda')
    env = NodeAttakEnv(vocab, relations, 70000, 0.94, 0.6075, delayed_reward_step, graph, init_avg_prob)
    dqn = DqnAgent( graph=graph, learning_rate=args.lr, num_nodes=num_nodes, num_relations=num_relations,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay_steps=args.epsilon_decay_steps,
        target_update=None,node_pre_embedding=node_pre_embedding,relation_pre_embedding=relation_pre_embedding,vocab_index=vocab_index,device=device,delayed_reward_step=delayed_reward_step)

    num_steps =int(70000)
    print('num_steps=',num_steps)
    state = [graph,[],0]

    ## start training
    ti.enable(args.enable_TimeTickTock)
    ti.tick()
    if args.mode_type=='train':
        for j in range(args.num_epochs):
            for i in tqdm(range(num_steps)):

                #action0 = random.randint(0,num_nodes-1)
                #while not action0 in graph:
                #    action0 = random.randint(0,num_nodes-1)

                action0 = random.randint(0,num_vocab_index-1)
                action0 = vocab_index[action0]
                #print('action0=',action0)
                action1 = dqn.sample_action1(state, action0)
                ti.tock('action choosing1')
                #print('action1=', action1)
                action2 = dqn.sample_action2(state, action0, action1)
                #print('action2=',action2)
                ti.tock('action choosing2')
                action=(action0, action1, {'rel':action2,'weight':1.0})
                print('action=',action)
                next_state, reward, terminal, score = env.step(action, state, model = model)
                ti.tock('step taken')
                dqn.store(state, action, reward, next_state, terminal, eval=False, curr_reward=False)
                ti.tock('store')
                loss=dqn.update()
                ti.tock('DQN update')
                ti.show()
                ti.reset()
                state=next_state
                #print('epoch:',j,', step:', i, 'reward=',reward, 'loss=',loss)
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{},{}\n'.format(i, reward, loss, score))

                if i%args.save_model_step ==0:
                    dqn.save(args.save_dir,j*args.num_epochs+i)
            env.reset()

        if args.num_epochs ==0:
            dqn.save(args.save_dir,0)
    elif args.mode_type=='eval':
        if args.num_epochs ==0:
            dqn.load(args.save_dir,0)
        else:
            dqn.load(args.save_dir,args.num_epochs-1)
        #main1()
        #acc0 = main2(mode = 'eval')
        #print('init_acc=', acc0)
        for i in tqdm(range(num_steps)):

            #print('step:', i)
            action0 = random.randint(0,num_vocab_index-1)
            action0 = vocab_index[action0]
            action1 = dqn.best_action1(state, action0)[0]
            ti.tock('action choosing1')
            action2 = dqn.best_action2(state, action0, action1)[0]
            ti.tock('action choosing2')
            action=(action0, action1, {'rel':action2,'weight':1.0})
            next_state, reward, terminal, score = env.step(action, state, is_train = False, model = model)
            ti.tock('step taken')
            ti.show()
            ti.reset()
            state=next_state
            with open(log_path, 'a') as fout:
                fout.write('{},{},{},{}\n'.format(i, reward, 0, score[0]))
        # main1()
        # acc = main2(mode = 'eval')
        # print('init_acc=', acc0)
        # print('acc=', acc)




if __name__ == '__main__':
    ti=TimeTickTock()
    main()
