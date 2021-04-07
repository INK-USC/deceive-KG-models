import numpy as np
import random
from Replay_Memory import *
#import tensorflow as tf
import os
import torch
from torch import nn
import math
import env

class Q1(nn.Module):
    def __init__(self,args, graph, node_pre_embedding, device):
        super(Q1,self).__init__()
        self.args=args
        self.graph = graph
        self.D_in = node_pre_embedding.shape[-1]+1
        self.node_pre_embedding=node_pre_embedding
        self.hidden_dims1 = args.q1_hidden_dims1  # for action0
        self.hidden_dims2 = args.q1_hidden_dims2  #for action1
        self.hidden_dims3 = args.q1_hidden_dims3  # for final embadding of action0 after lstm
        self.embed_dim= self.hidden_dims2[-1]
        self.device=device
        assert self.hidden_dims3[-1]==self.hidden_dims2[-1], 'Q1: embed_dim not the same!'

        self.lstm_cell1=nn.LSTMCell(self.hidden_dims1[-1], args.lstm_hidden_dim)

        self.linears=[]
        d0=self.D_in
        for d in self.hidden_dims1[:-1]:
            self.linears.append(nn.Linear(d0,d))
            self.linears.append(nn.ReLU(inplace=True))
            d0=d
        self.linears.append(nn.Linear(d0,self.hidden_dims1[-1]))
        self.linears=nn.Sequential(*self.linears)

        self.linears2=[]
        d0=self.D_in
        for d in self.hidden_dims2[:-1]:
            self.linears2.append(nn.Linear(d0,d))
            self.linears2.append(nn.ReLU(inplace=True))
            d0=d
        self.linears2.append(nn.Linear(d0,self.hidden_dims2[-1]))
        self.linears2=nn.Sequential(*self.linears2)

        self.linears3=[]
        d0=args.lstm_hidden_dim
        for d in self.hidden_dims3[:-1]:
            self.linears3.append(nn.Linear(d0,d))
            self.linears3.append(nn.ReLU(inplace=True))
            d0=d
        self.linears3.append(nn.Linear(d0,self.hidden_dims3[-1]))
        self.linears3=nn.Sequential(*self.linears3)

    def forward(self,states,actions0):
        graph=states[0]
        neighbors=[list(graph.adj[x.item()]) for x in actions0]  #[[neighbor_index]*nei_num]*batch
        pre_embeddings=[torch.cat([self.node_pre_embedding[x],torch.ones(len(x),1).to(self.device)*(states[2][i]/self.args.delayed_reward_step)],-1) for i,x in enumerate(neighbors)] #[(nei_num,pre_emb_dim+1)]*batch
        graph_embeddings=[self.linears2(x) for x in pre_embeddings]  #[(nei_num,emb_dim)]*batch
        action_embeddings, hc=self.embed_actions0(states,actions0) #(batch,emb_dim)
        values=[torch.matmul(graph_embeddings[i],action_embeddings[i])/math.sqrt(self.embed_dim) for i in range(action_embeddings.shape[0])]  #[(nei_num,)]*batch

        return neighbors, values, hc


    def embed_actions0(self, states, actions0):
        pre_embeddings=self.node_pre_embedding[actions0]  #(batch,pre_emb_dim)
        step_embs=states[2]/self.args.delayed_reward_step   #(batch, )
        pre_embeddings=torch.cat([pre_embeddings, step_embs[:,None]],-1)  #(batch,pre_emb_dim+1)
        embeddings=self.linears(pre_embeddings)  #(batch,emb_dim)
        hc = self.lstm_cell1(embeddings, states[3])  #(batch,lstm_hidden_dim)
        embeddings=self.linears3(hc[0])  #(batch, emb_dim)
        return embeddings, hc

    def first_not_in(self,index,nei,reject_list):
        for i,ind in enumerate(index):
            if nei[ind] not in reject_list:
                return ind
        print('all in reject_list!!')

    def max(self,states,actions0, print_value=False, return_hc=False,  reject_lists=None):
        neighbors,values, hc=self.forward(states,actions0)
        if reject_lists is None:
            # print(values)
            # print('action0: ', actions0)
            index=[torch.argmax(v).item() for v in values]  # [1]*batch
            # except:
            #     print('actions0: ', actions0)
            #     print('length of neighnours: ', len(neighbors))
            #     print('neighbours: ', neighbors)
            #     print('values: ', values)
            index=[neighbors[i][x] for i,x in enumerate(index)]  # [1]*batch
            m=torch.stack([torch.max(v) for v in values])   #(batch,)
        else:
            index=[torch.argsort(v,descending=True) for v in values]  # [num_nei]*batch
            index=[self.first_not_in(ind,neighbors[i],reject_lists[i]) for i,ind in enumerate(index)]
            m=torch.stack([v[index[i]] for i,v in enumerate(values)])
            index=[neighbors[i][x] for i,x in enumerate(index)]  # [1]*batch
        if print_value:
            print('Q1=',values)

        if return_hc:
            return m,index,hc
        else:
            return m,index

    def copy(self, Q):
        self.load_state_dict(Q.state_dict())
        self.eval()


class Q2(nn.Module):
    def __init__(self, args, graph, vocab_index, node_pre_embedding, relation_pre_embedding, relation_num, device):
        super(Q2,self).__init__()
        self.args=args
        self.graph=graph
        self.vocab_index=vocab_index
        self.D_in = 2*node_pre_embedding.shape[-1]+1
        self.D_in2 = relation_pre_embedding.shape[-1] if args.model_id not in [4,5,6,7] else node_pre_embedding.shape[-1]
        self.node_pre_embedding=node_pre_embedding
        self.relation_pre_embedding=relation_pre_embedding
        self.relation_num= relation_num
        self.hidden_dims1 = args.q2_hidden_dims1 #before lstm, for (action0,action1)
        self.hidden_dims2 = args.q2_hidden_dims2  #for relation
        self.hidden_dims3 = args.q2_hidden_dims3  #after lstm
        self.embed_dim= self.hidden_dims2[-1]

        self.device=device
        assert self.hidden_dims3[-1]==self.hidden_dims2[-1], 'Q2: embed_dim not the same!'

        self.lstm_cell2=nn.LSTMCell(self.hidden_dims1[-1],args.lstm_hidden_dim)

        self.linears=[]
        d0 = self.D_in
        for d in self.hidden_dims1[:-1]:
            self.linears.append(nn.Linear(d0,d))
            self.linears.append(nn.ReLU(inplace=True))
            d0=d
        self.linears.append(nn.Linear(d0,self.hidden_dims1[-1]))
        self.linears=nn.Sequential(*self.linears)

        self.linears2=[]
        d0=self.D_in2
        for d in self.hidden_dims2[:-1]:
            self.linears2.append(nn.Linear(d0,d))
            self.linears2.append(nn.ReLU(inplace=True))
            d0=d
        self.linears2.append(nn.Linear(d0,self.hidden_dims2[-1]))
        self.linears2=nn.Sequential(*self.linears2)

        self.linears3=[]
        d0=args.lstm_hidden_dim
        for d in self.hidden_dims3[:-1]:
            self.linears3.append(nn.Linear(d0,d))
            self.linears3.append(nn.ReLU(inplace=True))
            d0=d
        self.linears3.append(nn.Linear(d0,self.hidden_dims3[-1]))
        self.linears3=nn.Sequential(*self.linears3)

    def forward(self,states, hc, actions0, actions1, candidate_lists=None):


        if self.args.model_id in [4]:
            action2_pre_embedding=self.node_pre_embedding[self.vocab_index]
        elif self.args.model_id in [5]:
            action2_pre_embedding=self.node_pre_embedding
        else:
            action2_pre_embedding=self.relation_pre_embedding   #(17. emb_dim)
        if candidate_lists is not None:
            action_embeddings,hc=self.embed_actions(states,hc, actions0, actions1)   #(batch, emb_dim),((batch,lstm_hiddem_dim),(batch,lstm_hiddem_dim))
            values=[]
            for i in range(len(candidate_lists)):
                action2_pre_embedding1=action2_pre_embedding[candidate_lists[i]]
                action2_embedding = self.linears2(action2_pre_embedding1) #(100. emb_dim)


                # print('a1,a2=',action_embeddings.shape, action2_embedding.shape)
                #print('action_embeddings.shape=',action_embeddings.shape)
                value=torch.matmul(action_embeddings[i,None,None,:],action2_embedding[:,:,None])/math.sqrt(self.embed_dim)   #( 100,1,1)
                #print('values=', values)
                values.append(value[:,0,0])   #[(100)*batch]
            values=torch.stack(values,0)
        #     print('can=', candidate_lists[0].shape)
        # print('a2=',action2_pre_embedding.shape)
        else:
            action2_embedding = self.linears2(action2_pre_embedding) #(17. emb_dim)

            action_embeddings,hc=self.embed_actions(states,hc, actions0, actions1)   #(batch, emb_dim),((batch,lstm_hiddem_dim),(batch,lstm_hiddem_dim))
            # print('a1,a2=',action_embeddings.shape, action2_embedding.shape)
            #print('action_embeddings.shape=',action_embeddings.shape)
            values=torch.matmul(action_embeddings[:,None,None,:],action2_embedding[:,:,None])/math.sqrt(self.embed_dim)   #(batch, 17,1,1)
            #print('values=', values)
            values = values[:,:,0,0]   #(batch, 17)
        return values, hc

    def embed_actions(self, states, hc, actions0,actions1):
        #print('actions0=',actions0,'actions1=',actions1)
        #relations=[self.graph.adj[actions0[i].item()][actions1[i].item()][0]['rel'] for i in range(actions0.shape[0])]
        #print('embed_actions:actions0=',actions0,',actions1=',actions1,',relations=',relations)
        pre_embeddings=torch.cat([self.node_pre_embedding[actions0],self.node_pre_embedding[actions1]],dim=-1)  #(batch, D_in-1)
        step_embs=states[2]/self.args.delayed_reward_step   #(batch, )
        pre_embeddings=torch.cat([pre_embeddings, step_embs[:,None]],-1)  #(batch,D_in)

        embeddings=self.linears(pre_embeddings)   #(batch, lstm_input_dim)
        hc = self.lstm_cell2(embeddings, hc)  # (batch, lstm_hiddem_dim)
        embeddings = self.linears3(hc[0])  # (batch, emb_dim)

        return embeddings, hc

    def max(self,states, hc, actions0, actions1,print_value=False,return_hc=False, candidate_lists=None):
        if self.args.model_id<=3:
            values, hc=self.forward(states,hc, actions0, actions1)
            # print(values)
            if candidate_lists is None:
                m,index=torch.max(values,-1)
            else:
                # print('candidate lists in max: ', candidate_lists)
                index=torch.zeros(values.shape[0])
                m=torch.zeros(values.shape[0])
                # print('values=',values)
                for i,li in enumerate(candidate_lists):
                    # print('i,li=',i,li)
                    # print(values[i])
                    # print(values[i][li])
                    m[i],ind=torch.max(values[i][li],dim=-1)
                    index[i]=int(li[ind.item()])
        else:
            values, hc=self.forward(states,hc, actions0, actions1, candidate_lists)
            m,index1=torch.max(values,dim=-1)
            index=torch.zeros(index1.shape[0])
            if self.args.model_id==4:

                for i,ind in enumerate(index1):

                    index[i]=self.vocab_index[ind.item()]
            if candidate_lists is not None:  # take each candidate index
                for i,li in enumerate(candidate_lists):
                    #print(i, li)
                    index[i]=int(li[index1[i].item()])

        if print_value:
            print('Q2=',values)

        if return_hc:
            return m,index,hc
        else:
            return m,index

    def copy(self, Q):
        self.load_state_dict(Q.state_dict())
        self.eval()


class DqnAgent(object):


    def __init__(self, args, graph, num_nodes, num_relations, node_pre_embedding,relation_pre_embedding,vocab_index,device,hc0,ti):

        self.args=args
        self.graph=graph
        if args.model_id in [4,5,6]:
            self.num_actions2 = len(vocab_index)
        else:
            self.num_actions2 = num_relations
        node_pre_embedding=torch.tensor(node_pre_embedding).to(device)
        relation_pre_embedding=torch.tensor(relation_pre_embedding).to(device)

        self.vocab_index=vocab_index
        self.num_vocab_index=len(vocab_index)
        self.device=device
        self.hc0=hc0
        self.ti=ti

        self.model_id=args.model_id

        self.policyQ1=Q1(args, graph, node_pre_embedding,device=device).to(device)
        self.targetQ1=Q1(args, graph, node_pre_embedding,device=device).to(device)
        self.targetQ1.copy(self.policyQ1)

        self.policyQ2=Q2(args, graph,vocab_index, node_pre_embedding, relation_pre_embedding, relation_num=num_relations, device=device).to(device)
        self.targetQ2=Q2(args, graph,vocab_index, node_pre_embedding, relation_pre_embedding, relation_num=num_relations, device=device).to(device)
        self.targetQ2.copy(self.policyQ2)

        params=list(self.policyQ1.parameters())+list(self.policyQ2.parameters())+hc0
        self.optimizer=torch.optim.Adam(params,lr=self.args.dqn_lr)

        self.epsilons = np.linspace(args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        self.epsilon_decay_steps = args.epsilon_decay_steps

        self.replay_buffer = ReplayBuffer(args, device, hc0)

        self.current_time_step = 0


    def sample_action1(self, state, action0, epsilon=None, reject_list=None):
        #print('model_id=',self.model_id)
        if self.model_id in [1,6]:
            self.current_time_step += 1
        actions0=torch.tensor([action0]).to(self.device)
        if reject_list is not None:
            index, hc=self.best_actions1(state, actions0, [reject_list])
        else:
            index, hc=self.best_actions1(state, actions0)
        index=index[0]
        #print('sample_action1:index=',index)
        if epsilon is None:
            epsilon = self.epsilons[min(self.current_time_step, self.epsilon_decay_steps - 1)]
        e = random.random()
        if e < epsilon:
            graph=state[0]
            neighbors=list(graph.adj[action0])
            i=random.randint(0, len(neighbors) - 1)
            if reject_list is not None:
                while neighbors[i] in reject_list:
                    i=random.randint(0, len(neighbors) - 1)
            return neighbors[i] , hc
        else:
            return index, hc

    def sample_action2(self, state, hc, action0, action1, epsilon=None,candidate_list=None):
        if self.model_id in [0,3,4,5,7]:
            self.current_time_step += 1
        if candidate_list is None:
            candidate_list=list(range(self.num_actions2))
            # print('candidate_list: ', candidate_list)
            # pass
        if self.model_id in [0,4]:
            actions0=torch.tensor([action0]).to(self.device)
            actions1=torch.tensor([action1]).to(self.device)
            index, hc=self.best_actions2(state,hc, actions0, actions1)
            index=index[0]
        elif self.model_id in [3,5]:
            actions0=torch.tensor([action0]).to(self.device)
            actions1=torch.tensor([action1]).to(self.device)
            index, hc=self.best_actions2(state,hc, actions0, actions1, candidate_lists=[candidate_list])
            index=index[0]
        elif self.model_id==7:
            return self.vocab_index[random.randint(0, self.num_actions2 - 1)], hc
        else:
            raise RuntimeError('undefined model id!1')

        if epsilon is None:
            epsilon = self.epsilons[min(self.current_time_step, self.epsilon_decay_steps - 1)]
        e = random.random()
        if e < epsilon:
            if self.model_id in [4,7]:
                return self.vocab_index[random.randint(0, self.num_actions2 - 1)], hc
            return candidate_list[random.randint(0, len(candidate_list) - 1)], hc
        else:

            return index, hc

    def best_actions1(self, states, actions0, reject_lists=None):

        _,index,hc=self.policyQ1.max(states,actions0, print_value=False, return_hc=True,reject_lists=reject_lists)

        return index, hc

    def best_actions2(self, states, hc, actions0, actions1, candidate_lists=None):

        _, index,hc=self.policyQ2.max(states,hc, actions0,actions1, print_value=False,return_hc=True, candidate_lists=candidate_lists)
        return index, hc

    def store(self, action, reward, next_state, terminal, eval=False, curr_reward=False):
        if not eval:
            self.replay_buffer.add(action, reward, next_state, terminal)

    def update(self, graph=None):
        # if self.model_id not in [4]:  # donot consider the graph
        #     graph=None
        if self.current_time_step <= self.args.replay_init_size:
            return 0,0
        # sample data, should already be on device
        # init_step: (batch,); init_hc:2-tuple, (batch, lstm_hiddem_dim) ; else: (lstm_len, batch)
        init_step, init_hc, seq_actions0, seq_actions1, seq_actions2, seq_rewards, seq_terminals, avg_abs_rewards = self.replay_buffer.sample()
        self.ti.tock('replay_memory_sampling',1)
        assert seq_actions0.shape[0]==self.args.dqn_lstm_len, 'dqn_lstm_len != seq_actions0.shape[0]'
        #print('batch_size=',seq_actions0.shape[1])
        loss=0
        hcs=[]  #should be lstm_len-length, for next_state
        hc = init_hc
        next_state= [graph,None,init_step, hc ]
        # losses=torch.zeros((seq_actions0.shape[1],),device=self.device)  #(batch, )
        loss=0
        for seq in range(seq_actions0.shape[0]):
            states=next_state

            actions0=seq_actions0[seq] #(batch, )
            actions1=seq_actions1[seq]
            actions2=seq_actions2[seq]
            rewards=seq_rewards[seq]
            terminals=seq_terminals[seq]

            nei, q_values1, hc = self.policyQ1(states,actions0)
            actions1_inside = [nei[i].index(actions1[i]) for i in range(len(nei))] #[1]*batch
            q_values_pred1 = [q_values1[i][actions1_inside[i]] for i in range(len(nei))] #[1]*batch
            q_values_pred1 = torch.stack(q_values_pred1,0)  #(batch, )

            if self.model_id in [0,3,4]:
                next_states_max_q_values2,_ = self.targetQ2.max(states,hc, actions0, actions1)
                td_targets1 =  next_states_max_q_values2        #(batch, )
            elif self.model_id in [5]:
                candidate_list=random.sample(self.vocab_index,100)
                next_states_max_q_values2,_ = self.targetQ2.max(states,hc, actions0, actions1,candidate_lists=[candidate_list])
                td_targets1 =  next_states_max_q_values2        #(batch, )

            if self.model_id in [0,3]:
                q_values2, hc = self.policyQ2(states,hc,actions0, actions1)
                q_values_pred2 = [q_values2[i][actions2[i]] for i in range(len(nei))]
                q_values_pred2 = torch.stack(q_values_pred2, 0)   #(batch, )
            elif self.model_id in [4]:
                q_values2, hc = self.policyQ2(states,hc,actions0, actions1)
                # print(actions2)
                # for i in range(len(nei)):
                    # print(actions0[i],actions2[i])
                    # print(self.vocab_index.index(actions2[i].item()))
                q_values_pred2 = [q_values2[i][self.vocab_index.index(actions2[i].item())] for i in range(len(nei))]
                q_values_pred2 = torch.stack(q_values_pred2, 0)   #(batch, )
            elif self.model_id in [5]:
                candidate_lists=actions2[:,None]
                q_values2, hc = self.policyQ2(states,hc,actions0, actions1,candidate_lists=candidate_lists)

                q_values_pred2 = q_values2[:,0]
                # q_values_pred2 = torch.stack(q_values_pred2, 0)   #(batch, )


            next_states=[graph,None,states[2]+1,hc]
            # hcs.append([hc[0].detach().clone(),hc[1].detach().clone()])
            hcs.append([hc[0].detach(),hc[1].detach()])


            next_actions0 = torch.tensor([self.vocab_index[random.randint(0,self.num_vocab_index - 1)] for i in range(len(actions0))]).to(self.device)  #(batch, )
            #print(next_actions0, 'next_actions0.shape=', next_actions0.shape)
            next_states_max_q_values1,_ = self.targetQ1.max(next_states,next_actions0)    #(batch, )
            #print('next_states_max_q_values1.shape=',next_states_max_q_values1.shape)
            td_targets2 =rewards + (1 - terminals) * self.args.dqn_discount * next_states_max_q_values1   #(batch, )

            if self.model_id in [0,3,4,5]:
                #print('0000000000')
                loss += torch.mean(clipped_error(q_values_pred1 - td_targets1))+torch.mean(clipped_error(q_values_pred2 - td_targets2))  #(batch, )
                #losses +=clipped_error(q_values_pred1 - td_targets1)+clipped_error(q_values_pred2 - td_targets2)  #(batch, )
            elif self.model_id in [1,6]:
                loss += torch.mean(clipped_error(q_values_pred1 - td_targets2))
                # losses += clipped_error(q_values_pred1 - td_targets2)  #(batch, )
        self.ti.tock('forward_compute',1)
        # loss = torch.mean(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ti.tock('backward_compute',1)
        self.replay_buffer.update(hcs)
        self.ti.tock('replay_memory_update',1)

        # Update the target q-network.
        if not self.current_time_step % self.args.dqn_target_update:
            self.targetQ1.copy(self.policyQ1)
            self.targetQ2.copy(self.policyQ2)
            print('Target updated!')
            self.ti.tock('target_update',1)
        print('loss=',loss)
        return loss, avg_abs_rewards


    def save(self, save_dir, epoch):
        #saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path=os.path.join(save_dir,'model_'+str(epoch)+'.ckpt')
        print(path)
        torch.save([self.policyQ1.state_dict(), self.policyQ2.state_dict()],path)
        path=os.path.join(save_dir,'data.pkl')
        print('saved model!')



    def load(self, load_dir, epoch):
        path=os.path.join(load_dir,'model_'+str(epoch)+'.ckpt')
        data = torch.load(path)
        self.policyQ1.load_state_dict(data[0])
        self.policyQ2.load_state_dict(data[1])
        self.targetQ1.copy(self.policyQ1)
        self.targetQ2.copy(self.policyQ2)

    def save_data(self,save_dir):
        path=os.path.join(save_dir,'data.pkl')
        self.replay_buffer.save(path)
        print('saved data!')



def clipped_error(x):
    return torch.where(torch.abs(x) < 1.0, 0.5 * x*x, torch.abs(x) - 0.5)
