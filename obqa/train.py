
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
# from test_script import *
from utils_time import *
# 60.5 without KG




def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--enable_TimeTickTock', action='store_true',default=False)
    parser.add_argument('--debug_mode', action='store_true',default=False)
    parser.add_argument('--no_cuda', action='store_true',default=False)
    parser.add_argument('--avoid_same_at_training', action='store_true',default=False)
    parser.add_argument('--backup_graph_step', type=int,default=0)
    parser.add_argument('--backup_graph_name', type=str,default='graph.graph')
    #parameters for dqn network
    parser.add_argument('--lstm_hidden_dim',default=1000, type=int, help='dqn_lstm_hidden_dim.')
    parser.add_argument('--q1_hidden_dims1',default=[128,128], nargs='*', type=int, help='Embed current action before lstm.')
    parser.add_argument('--q1_hidden_dims2',default=[128,128], nargs='*', type=int, help='Embed target')
    parser.add_argument('--q1_hidden_dims3',default=[128], nargs='*', type=int, help='Embed current action after lstm')
    parser.add_argument('--q2_hidden_dims1',default=[128,128], nargs='*', type=int, help='Embed current action before lstm.')
    parser.add_argument('--q2_hidden_dims2',default=[128,128], nargs='*', type=int, help='Embed target')
    parser.add_argument('--q2_hidden_dims3',default=[128], nargs='*', type=int, help='Embed current action after lstm')

    #parameters for dqn
    parser.add_argument('--model_id', default=0,type=int, help='model id,different for different models. 0-use 2 DQNs; 1-use 1 DQN+least relation; ')
    parser.add_argument('--graph_model', default=0, type=int, help='model id=0:RN,  1:GN')
    parser.add_argument('--dqn_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--dqn_discount',default=0.99999, type=float, help='discount of DQN')
    parser.add_argument('--dqn_target_update',default=100, type=int, help='Update the target network every TARGET_UPDATE timesteps.')
    parser.add_argument('--least_relation_num',default=5, type=int, help='least k relations to choose')


    parser.add_argument('--epsilon_start', default=1.0, type=float, help='epsilon_start')
    parser.add_argument('--epsilon_end', default=0.01, type=float, help='epsilon_end')
    parser.add_argument('--epsilon_decay_steps', default=5000, type=int, help='epsilon_decay_steps')

    #  Replay_buffer
    parser.add_argument('--dqn_lstm_len',default=50, type=int, help='length of lstm during training')
    parser.add_argument('--replay_memory_size',default=400000, type=int, help='Max size of the replay buffer.')
    parser.add_argument('--replay_init_size',default=1000, type=int, help='Initial size of replay memory prior to beginning sampling batches.')
    parser.add_argument('--enable_shuffle', action='store_true', default=False, help='whether to shuffle the order of actions during training')
    parser.add_argument('--dqn_shuffle_rate',default=1.0, type=float, help='prob to shuffle.')
    parser.add_argument('--reward_expectation',default=5.0, type=float, help='reward_expectation.')

    #parameters for Training
    #parser.add_argument('--pertub_rate', default=0.5, type=float, help='rate of pertubation')
    #parser.add_argument('--update_rate', default = 10, type = int)
    parser.add_argument('--dqn_train_step',default=1, type=int, help='Batch size for updates from the replay buffer.')
    parser.add_argument('--delayed_reward_step',default=500, type=int, help='Batch size for updates from the replay buffer.')
    parser.add_argument('--mode_type', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--num_steps',default=30001, type=int, help='total num of steps per epoch')
    parser.add_argument('--steps_after_collecting_data',default=1000, type=int, help='total num of steps per epoch')
    parser.add_argument('--dqn_batch_size',default=32, type=int, help='Batch size for updates from the replay buffer.')
    parser.add_argument('--save_model_step', default = 5000, type = int)

    #  save_paths, log_path
    parser.add_argument('--save_dir', default=f'./saved_models/KG/', help='model output directory')
    parser.add_argument('--log_path', default=f'log.csv', help='path for log')

    #  load_paths
    parser.add_argument('--classifier_path', default='./deep_classifier_1.hdf5')
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
    parser.add_argument('--cpnet_index', default='./new_vocab_2.txt')
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

    #parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    #parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    args = parser.parse_args()
    train(args)


def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

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
    num_vocab_index=len(vocab_index)
    my_file.close()


    with open(args.log_path, 'w') as fout:
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

    if args.debug_mode:
        init_prob_list=np.array([0]*400)
        init_acc_list=np.array([0]*400)
        init_avg_prob=0.55021113
    else:
        if args.graph_model==0:
            main1()
            init_prob_list,init_acc_list = main2(mode = 'eval')
        elif args.graph_model==1:
            main3()
            init_prob_list,init_acc_list = main4(mode = 'eval')
        elif args.graph_model==2:
            main5()
            init_prob_list,init_acc_list = main6(mode = 'eval')
        else:
            print('Undefined graph_model!',args.graph_model)
            exit()
        init_avg_prob = get_prob_cost(init_prob_list)
        init_acc = np.mean(init_acc_list)
        print('init_prob_cost=',init_avg_prob,'init_acc=', init_acc)

    num_nodes = len(vocab)
    num_relations = len(relations)
    print('num_nodes=',num_nodes, ', num_vocab_index=',num_vocab_index, ', num_relations=', num_relations)

    node_pre_embedding = np.load(args.cpt_emb)
    relation_pre_embedding = np.load(args.relation_emb)
    print('node_pre_embedding=',node_pre_embedding.shape, ', relation_pre_embedding=', relation_pre_embedding.shape)

    if args.no_cuda:
        device=torch.device('cpu')
    else:
        device=torch.device('cuda')

    h0=torch.randn(1,args.lstm_hidden_dim,requires_grad=True, device=device)
    c0=torch.randn(1,args.lstm_hidden_dim,requires_grad=True, device=device)
    hc0=[h0,c0]
    print('0: hc0_is_leaf=', hc0[0].is_leaf, hc0[1].is_leaf)
    hc0_detach=[h0.detach(),c0.detach()]

    # model=load_model()
    model = None

    env = NodeAttakEnv(args, vocab_index, vocab, relations, 70000, 0.94, 0.6075, args.delayed_reward_step, graph, init_avg_prob, hc0_detach, device)
    dqn = DqnAgent( args=args,  graph=graph, num_nodes=num_nodes, num_relations=num_relations,
            node_pre_embedding=node_pre_embedding,relation_pre_embedding=relation_pre_embedding,vocab_index=vocab_index,device=device,
            hc0= hc0 ,ti=ti)

    ## start training
    ti.enable(args.enable_TimeTickTock)
    ti.tick()
    with torch.autograd.set_detect_anomaly(True):
        state = [graph,[], torch.tensor([0],dtype=torch.float).to(device), hc0_detach]   # initial state
        if args.mode_type=='train':
            for j in range(args.num_epochs):
                pre_actions={}
                for i in vocab_index:
                    pre_actions[i]=[]
                for i in tqdm(range(args.num_steps)):

                    # action0 = random.randint(0,num_vocab_index-1)
                    # action0 = vocab_index[action0]

                    action0 = get_random_action0(state[0],num_vocab_index,vocab_index)
                    if args.model_id<=3 and args.avoid_same_at_training:
                        while len(pre_actions[action0])==len(graph[action0]):
                            action0 = random.randint(0,num_vocab_index-1)
                            action0 = vocab_index[action0]
                    #print('action0=',action0)
                    with torch.no_grad():
                        if args.avoid_same_at_training:
                            action1, hc = dqn.sample_action1(state, action0,reject_list=pre_actions[action0])
                        else:
                            action1, hc = dqn.sample_action1(state, action0)
                        ti.tock('action choosing1')
                        #print('action1=', action1)
                        if args.model_id==1:  #least
                            score=env.predict_score(np.array([action0]),np.array([action1]))
                            # print('action=',action0,action1)
                            # print('score0=',score)
                            score=score[0]
                            action2 = np.argmin(score)
                            # print('score=',score,np.argsort(score))
                            # print('action2=',action2)

                        elif args.model_id in [0,4]:
                            action2, hc = dqn.sample_action2(state, hc, action0, action1)
                        elif args.model_id in [5]:
                            G=state[0]
                            idx=len(G[action0][action1])-1
                            candidate_list=env.least_edge_val(action0, G[action0][action1][idx]['rel'])
                            print(candidate_list.shape)
                            action2, hc = dqn.sample_action2(state, hc, action0, action1,candidate_list=candidate_list )


                        elif args.model_id in [6]: # choose the least val_score
                            G=state[0]
                            idx=len(G[action0][action1])-1
                            action2 = env.least_edge_val(action0, G[action0][action1][idx]['rel'])

                        elif args.model_id==3:  #least k
                            score=env.predict_score(np.array([action0]),np.array([action1]))
                            # print('action=',action0,action1)
                            # print('score0=',score)
                            score=score[0]
                            candidate_list = np.argsort(score)[0:args.least_relation_num]
                            # print('score=',score,np.argsort(score))
                            # print(candidate_list)
                            action2,hc = dqn.sample_action2(state, hc, action0, action1, candidate_list=candidate_list)

                        else:
                            print('Undefined model_id!',args.model_id)
                            exit()
                    #print('action2=',action2)
                        ti.tock('action choosing2')
                    action=(action0, action1, {'rel':action2,'weight':1.0})
                    print('action=',action)
                    pre_actions[action0].append(action1)
                    next_state, reward, terminal, score, acc, avg_prob, prob_list,graph2 = env.step(action, state, hc, model)
                    ti.tock('step taken')
                    dqn.store(action, reward, next_state, terminal, eval=False, curr_reward=False)
                    ti.tock('store')
                    if i%args.dqn_train_step==0:
                        loss,avg_abs_rewards=dqn.update(graph2)
                    else:
                        loss=0
                        avg_abs_rewards=0
                    ti.tock('DQN update')
                    ti.show()
                    ti.reset()
                    state=next_state
                    #print('epoch:',j,', step:', i, 'reward=',reward, 'loss=',loss)
                    with open(args.log_path, 'a') as fout:
                        fout.write('{},{},{},{},{},{},{},{},{}\n'.format(i, reward, loss,avg_abs_rewards, score,acc,avg_prob,action, prob_list))
                    with open(args.log_path+'.pkl', 'ab') as fout:
                        pickle.dump([i, reward, loss,avg_abs_rewards, score,acc,avg_prob,action, prob_list], fout)

                    if i%args.save_model_step ==0:
                        dqn.save(args.save_dir,j*args.num_epochs+i)
                env.reset()

            for i in range(args.steps_after_collecting_data+1): #Training
                loss,avg_abs_rewards=dqn.update(graph2)
                dqn.current_time_step+=1
                with open(args.log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(i, loss, avg_abs_rewards))
                with open(args.log_path+'.pkl', 'ab') as fout:
                    pickle.dump([i, loss,avg_abs_rewards], fout)
                if i%200==0:
                    dqn.save(args.save_dir,70000+i)
            # dqn.save_data(args.save_dir)
            if args.num_epochs ==0:
                dqn.save(args.save_dir,0)
                dqn.save_data(args.save_dir)
        elif args.mode_type=='eval':
            env.evaluate()
            if args.num_epochs ==0:
                dqn.load(args.save_dir,0)
            else:
                dqn.load(args.save_dir,args.num_epochs-1)

            acc0=np.mean(init_acc_list)
            print('init_acc=', acc0)
            pre_actions={}
            for i in vocab_index:
                pre_actions[i]=[]
            for i in tqdm(range(args.num_steps)):
                if args.backup_graph_step>0 and (i+1)%args.backup_graph_step==0:
                    with open('./RL_saved_graph/{}_{}.graph'.format(args.backup_graph_name,i+1),'wb') as f:
                        pickle.dump(state[0],f)
                action0 = get_random_action0(state[0],num_vocab_index,vocab_index)
                if args.model_id<=3:
                    while len(pre_actions[action0])==len(graph[action0]):
                        action0 = random.randint(0,num_vocab_index-1)
                        action0 = vocab_index[action0]

                with torch.no_grad():
                    if args.model_id==2:
                        action1, hc = dqn.sample_action1(state, action0,epsilon=2,reject_list=pre_actions[action0])
                    elif args.model_id<=3:
                        action1, hc = dqn.sample_action1(state, action0,epsilon=0,reject_list=pre_actions[action0])
                    elif args.model_id==7:
                        action1, hc = dqn.sample_action1(state, action0,epsilon=2)
                    else:
                        action1, hc = dqn.sample_action1(state, action0,epsilon=0)
                    ti.tock('action choosing1')

                    if args.model_id in [1,2]:
                        score=env.predict_score(np.array([action0]),np.array([action1]))[0]
                        action2 = np.argmin(score)
                    elif args.model_id in [0,4]:
                        action2,hc = dqn.sample_action2(state, hc, action0, action1,epsilon=0)
                    elif args.model_id==3:
                        score=env.predict_score(np.array([action0]),np.array([action1]))[0]
                        candidate_list = np.argsort(score)[0:args.least_relation_num]
                        action2,hc = dqn.sample_action2(state, hc, action0, action1, candidate_list=candidate_list)
                    elif args.model_id==5:
                        G=state[0]
                        idx=len(G[action0][action1])-1
                        candidate_list=env.least_edge_val(action0, G[action0][action1][idx]['rel'])
                        # print(candidate_list.shape)
                        action2, hc = dqn.sample_action2(state, hc, action0, action1,epsilon=0,candidate_list=candidate_list )
                    elif args.model_id in [6]: # choose the least val_score
                        G=state[0]
                        idx=len(G[action0][action1])-1
                        action2 = env.least_edge_val(action0, G[action0][action1][idx]['rel'])
                    elif args.model_id in [7]:
                        action2, hc = dqn.sample_action2(state, hc, action0, action1,epsilon=2)
                    else:
                        print('Undefined model_id!')

                    ti.tock('action choosing2')
                    action=(action0, action1, {'rel':action2,'weight':1.0})
                    pre_actions[action0].append(action1)
                    next_state, reward, terminal, score, acc, avg_prob, prob_list,graph2 = env.step(action, state, hc, model)
                    ti.tock('step taken')
                    ti.show()
                    ti.reset()
                    state=next_state
                    loss=0
                    with open(args.log_path, 'a') as fout:
                        fout.write('{},{},{},{},{},{}\n'.format(i, reward, loss, score,acc, action))

            #main1()
            #acc = main2(mode = 'eval')
            with open('./RL_saved_graph/{}_final.graph'.format(args.backup_graph_name),'wb') as f:
                pickle.dump(state[0],f)
            acc=0
            print('init_acc=', acc0)
            print('acc=', acc)




if __name__ == '__main__':
    ti=TimeTickTock()
    main()
