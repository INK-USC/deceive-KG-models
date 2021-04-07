import os
import numpy as np
import pickle
from tqdm import tqdm
import torch

class Data_Helper(object):
    """docstring for Data_Helper"""
    def __init__(self, dataset):
        super().__init__()

        data_path = os.path.join('./data', dataset, 'kg_final.npy')
        kg = np.load(data_path)    
        self.get_kg_stat(kg)
        
        print('splitting dataset ...')

        # train:eval:test = 8:1:1
        if dataset == 'music':
            eval_num = 500
            test_num = 500
        elif dataset == 'movie':
            eval_num = 2500
            test_num = 2500
        # eval_ratio = 0.1
        # test_ratio = 0.1
        n_triples = kg.shape[0]

        eval_indices = np.random.choice(n_triples, size=eval_num, replace=False)
        left = set(range(n_triples)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=test_num, replace=False)
        train_indices = list(left - set(test_indices))

        self.train_data = kg[train_indices]
        self.eval_data = kg[eval_indices]
        self.test_data = kg[test_indices]

        self.train_pool = set([tuple(triple) for triple in self.train_data])
        self.eval_pool = set([tuple(triple) for triple in self.eval_data])
        self.test_pool = set([tuple(triple) for triple in self.test_data])

        self.golden_triple_pool = self.train_pool | self.eval_pool | self.test_pool 

    def get_kg_stat(self, kg):
        entity_set = set()
        relation_set = set()
        for triple in kg:
            head, relation, tail = triple
            entity_set.add(head)
            entity_set.add(tail)
            relation_set.add(relation)

        self.n_entity = len(entity_set)
        self.n_relation = len(relation_set)
        self.entities = list(entity_set)
        self.relations = list(relation_set)
        print('#entity: {}'.format(self.n_entity))
        print('#relation: {}'.format(self.n_relation))

        self.relation_stat_dict = {}
        for relation_type in relation_set:
            group_head_list = [triple[0] for triple in kg if triple[1] == relation_type]
            group_tail_list = [triple[2] for triple in kg if triple[1] == relation_type]
            head_per_tail = len(group_head_list) * 1.0 / len(set(group_tail_list))
            tail_per_head = len(group_tail_list) * 1.0 / len(set(group_head_list))
            self.relation_stat_dict[relation_type] = (head_per_tail, tail_per_head)

    def get_next_batch(self, dataset, start, end, device, num_negative=0):
        batch_pos = dataset[start:end]
        batch_data = batch_pos
        batch_label = np.ones(len(batch_pos))
        for _iter in range(num_negative):
            batch_neg = []
            for head, relation, tail in batch_pos:
                head_neg = head
                tail_neg = tail
                # corrupt_head_prob = np.random.binomial(1, 0.5)
                hpt, tph = self.relation_stat_dict[relation]
                corrupt_head_prob = np.random.binomial(1, (tph / (tph + hpt)))
                while True:
                    if corrupt_head_prob:
                        head_neg = np.random.choice(self.entities)
                    else:
                        tail_neg = np.random.choice(self.entities)
                    if (head_neg, relation, tail_neg) not in self.train_pool:
                        break
                batch_neg.append([head_neg, relation, tail_neg])
            batch_data = np.append(batch_data, batch_neg, 0)
            batch_label = np.append(batch_label, np.zeros(len(batch_neg)), 0)

        batch_data = torch.LongTensor(batch_data).to(device)
        batch_label = torch.FloatTensor(batch_label).to(device)

        return batch_data, batch_label

    def get_eval_batch(self, triple, device):
        head, relation, tail = triple

        batch_head_input = [[head,relation,tail]]
        batch_head_input.extend([[head_neg, relation, tail] for head_neg in self.entities if not (head_neg, relation, tail) in self.golden_triple_pool])

        batch_tail_input = [[head,relation,tail]]
        batch_tail_input.extend([[head, relation, tail_neg] for tail_neg in self.entities if not (head, relation, tail_neg) in self.golden_triple_pool])

        batch_head_input = torch.LongTensor(batch_head_input).to(device)
        batch_tail_input = torch.LongTensor(batch_tail_input).to(device)
        return batch_head_input, batch_tail_input


