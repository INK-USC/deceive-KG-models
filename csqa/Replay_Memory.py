import numpy as np
import random
import pickle
import torch
from utils_time import *

class ReplayBuffer(object):

    def __init__(self, args, device, hc0):
        self.args=args
        self.device=device
        self.h0 = hc0[0]  #(1,lstm_hiddem_dim)
        self.c0 = hc0[1]  #(1,lstm_hiddem_dim)
        #self.max_total_size = args.replay_memory_size
        #self.init_size = args.replay_init_size
        #self.batch_size = args.dqn_batch_size

        #self.delayed_reward_step=args.delayed_reward_step
        self.max_row_size =args.replay_memory_size//args.delayed_reward_step
        self.sum_abs_rewards=0
        self.num_rewards=0

        self.reach_max=False
        self.curr_row = 0      #0 - (self.max_size-1)
        self.curr_pointer = 0   #0-499, the next adding

        self.hs = []
        self.cs = []
        self.actions0 = []
        self.actions1 = []
        self.actions2 = []
        self.rewards = []
        self.terminals = []

        self.add_row()

    def add_row(self):
        args=self.args
        device=self.device
        if self.reach_max:
            return
        self.hs.append([None]*args.delayed_reward_step)
        self.cs.append([None]*args.delayed_reward_step)
        self.hs[-1][0]=self.h0
        self.cs[-1][0]=self.c0
        self.actions0.append(torch.zeros(args.delayed_reward_step,dtype=torch.long).to(device))
        self.actions1.append(torch.zeros(args.delayed_reward_step,dtype=torch.long).to(device))
        self.actions2.append(torch.zeros(args.delayed_reward_step,dtype=torch.long).to(device))
        self.rewards.append(torch.zeros(args.delayed_reward_step).to(device))
        self.terminals.append(torch.zeros(args.delayed_reward_step).to(device))

    def add(self, action, reward, next_state, terminal):

        #self.states[self.curr_row][self.curr_pointer] = state
        self.actions0[self.curr_row][self.curr_pointer] = action[0]
        self.actions1[self.curr_row][self.curr_pointer] = action[1]
        self.actions2[self.curr_row][self.curr_pointer] = action[2]['rel']
        self.rewards[self.curr_row][self.curr_pointer] = reward
        self.terminals[self.curr_row][self.curr_pointer] = terminal

        self.curr_pointer += 1
        if self.curr_pointer>= self.args.delayed_reward_step:  #new row
            print('reward=',reward)
            assert terminal==1, 'terminal is not 1 at the end, {}'.format(terminal)
            self.curr_pointer = 0
            self.num_rewards+=1
            self.sum_abs_rewards+=abs(reward)
            print('self.sum_abs_rewards=',self.sum_abs_rewards,', self.num_rewards=',self.num_rewards,'avg=',self.sum_abs_rewards/self.num_rewards)
            if self.curr_row == self.max_row_size-1:
                self.reach_max=True
            self.curr_row = (self.curr_row +1)% self.max_row_size
            self.add_row()
            self.hs[self.curr_row][self.curr_pointer]=self.h0
            self.cs[self.curr_row][self.curr_pointer]=self.c0
        else:
            self.hs[self.curr_row][self.curr_pointer]=next_state[3][0]
            self.cs[self.curr_row][self.curr_pointer]=next_state[3][1]


    def swap(self,arr,row,pos,pos2):
        t=arr[row][pos2].detach().clone()
        arr[row][pos2] = arr[row][pos]
        arr[row][pos] = t

    def shuffle(self,row,pos):
        for i in range(pos,pos+self.args.dqn_lstm_len):
            pos2=random.randint(i,pos+self.args.dqn_lstm_len-1)
            # no need to shuffle hs and cs
            self.swap(self.actions0,row,pos,pos2)
            self.swap(self.actions1,row,pos,pos2)
            self.swap(self.actions2,row,pos,pos2)
            # self.swap(self.rewards,row,pos,pos2)
            # self.swap(self.terminals,row,pos,pos2)




    def sample(self):
        if self.reach_max:
            rows_for_selection=list(range(self.curr_row))+list(range(self.curr_row+1,self.max_row_size))
        else:
            rows_for_selection=list(range(self.curr_row))
        len_selection=len(rows_for_selection)
        
        positions=[]
        for i in range(len_selection):
            positions.append([])
        cnt=0
        for i in range(self.args.dqn_batch_size*2):
            row = random.randint(0,len_selection-1)
            pos = random.randint(0,self.args.delayed_reward_step-1)  #begin pos
            if pos>=self.args.delayed_reward_step-self.args.dqn_lstm_len//2:
                pos=0
            elif pos>=self.args.delayed_reward_step-self.args.dqn_lstm_len:
                pos=self.args.delayed_reward_step-self.args.dqn_lstm_len
            flag=True
            for j in positions[row]:
                if abs(pos-j)<self.args.dqn_lstm_len:
                    flag=False
            if flag:
                positions[row].append(pos)
                if self.args.enable_shuffle:
                    if random.random()<self.args.dqn_shuffle_rate:
                        self.shuffle(rows_for_selection[row],pos)
                cnt+=1
                if cnt>=self.args.dqn_batch_size:
                    break
        # print('rows_for_selection=',rows_for_selection)
        # print('positions=',positions)
        # print('cnt=', cnt)
        self.rows_for_selection=rows_for_selection
        self.positions=positions
        steps=[]
        hs = []
        cs = []
        seq_actions0 = []
        seq_actions1 = []
        seq_actions2 = []
        seq_rewards = []
        seq_terminals = []

        cnt=0

        for i,poss in enumerate(positions):
            row=rows_for_selection[i]
            for pos in poss:
                cnt+=1
                steps.append(pos)
                hs.append(self.hs[row][pos])
                cs.append(self.cs[row][pos])
                seq_actions0.append(self.actions0[row][pos:pos+self.args.dqn_lstm_len])
                seq_actions1.append(self.actions1[row][pos:pos+self.args.dqn_lstm_len])
                seq_actions2.append(self.actions2[row][pos:pos+self.args.dqn_lstm_len])
                seq_rewards.append(self.rewards[row][pos:pos+self.args.dqn_lstm_len])
                seq_terminals.append(self.terminals[row][pos:pos+self.args.dqn_lstm_len])
        # print('cnt2=',cnt)
        steps=torch.tensor(steps,dtype=torch.float,device=self.device)
        hs=torch.cat(hs,0)
        cs=torch.cat(cs,0)
        seq_actions0=torch.stack(seq_actions0,-1)
        seq_actions1=torch.stack(seq_actions1,-1)
        seq_actions2=torch.stack(seq_actions2,-1)
        seq_rewards=torch.stack(seq_rewards,-1)
        print('scale=',(self.args.reward_expectation/(self.sum_abs_rewards/self.num_rewards+1e-10)))
        seq_rewards=seq_rewards*(self.args.reward_expectation/(self.sum_abs_rewards/self.num_rewards+1e-10))
        seq_terminals=torch.stack(seq_terminals,-1)
        # print(('scale=', self.args.reward_expectation/(self.sum_abs_rewards/self.num_rewards+1e-10)))
        # print('seq_rewards=',seq_rewards)
        # print('seq_terminals=', seq_terminals)
        # print('seq_actions0=',seq_actions0)
        return steps,[hs,cs],seq_actions0, seq_actions1, seq_actions2, seq_rewards, seq_terminals, self.sum_abs_rewards/self.num_rewards

    def update(self, hcs):
        cnt=0
        for i,poss in enumerate(self.positions):
            row=self.rows_for_selection[i]
            for pos in poss:
                for j in range(pos+1,pos+1+self.args.dqn_lstm_len):
                    if j==self.args.delayed_reward_step:
                        continue
                    self.hs[row][j]=hcs[j-pos-1][0][cnt:(cnt+1)]
                    self.cs[row][j]=hcs[j-pos-1][1][cnt:(cnt+1)]
                cnt+=1


    def save(self,filename):
        data=[self.hs, self.cs, self.actions0, self.actions1, self.actions2, self.rewards, self.terminals]
        state=[self.args, self.device, self.h0 ,self.c0 ,self.max_row_size ,self.reach_max, self.curr_row ,self.curr_pointer]   #0-499, the next adding]
        with open(filename,'wb') as f:
            pickle.dump([state,data],f)

    def load(self,filenname):
        data=None
        with open(filename,'rb') as f:
            state,data=pickle.load(f)
        self.args, self.device, self.h0 ,self.c0 ,self.max_row_size ,self.reach_max, self.curr_row ,self.curr_pointer =state
        self.hs, self.cs, self.actions0, self.actions1, self.actions2, self.rewards, self.terminals=data
        return self.h0,self.c0
