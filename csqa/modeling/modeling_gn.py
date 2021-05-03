from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
# def create_dataset(adj_path):
#     with open(adj_path, 'rb') as handle:
#         data = pickle.load(handle)
#data.x = c_ids, data.edge_index = edge_index, data.edge_attr = rel_ids, y= num_tuples
def open_graph(graph_path):
    with open(graph_path, 'rb') as handle:
        graph_list = pickle.load(handle)
    data_list = []
    num_tuples = []
    for graph in tqdm(graph_list):
        row = graph[0].row
        col = graph[0].col
        edge_attr = torch.tensor(graph[0].data)
        edge_index = torch.tensor(np.concatenate((row.reshape(1,row.shape[0]), col.reshape(1, col.shape[0])),axis = 0), dtype=torch.long)
        x = torch.tensor(graph[1])
        y = torch.tensor([x.shape[0]])
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(d)
        #num_tuples.append(row.shape[0])
    return data_list

class GraphDataLoader(object):
    def __init__(self, train_adj_path, dev_adj_path, test_adj_path, batch_size, eval_batch_size):
        self.batch_size = 5*batch_size
        self.eval_batch_size = 5*eval_batch_size
        self.train_graph = open_graph(train_adj_path)
        self.dev_graph = open_graph(dev_adj_path)
        self.test_graph = open_graph(test_adj_path)

    def train_graph_data(self,train_indexes):
        train_data = []
        for i in train_indexes:
            train_data.append(self.train_graph[5*i])
            train_data.append(self.train_graph[5*i + 1])
            train_data.append(self.train_graph[5*i + 2])
            train_data.append(self.train_graph[5*i + 3])
            train_data.append(self.train_graph[5*i + 4])

        return DataLoader(train_data, batch_size=self.batch_size)

    def dev_graph_data(self):
        return DataLoader(self.dev_graph, batch_size=self.eval_batch_size)

    def test_graph_data(self,test_indexes):
        test_data = []
        for i in test_indexes:
            test_data.append(self.train_graph[5*i])
            test_data.append(self.train_graph[5*i + 1])
            test_data.append(self.train_graph[5*i + 2])
            test_data.append(self.train_graph[5*i + 3])
            test_data.append(self.train_graph[5*i + 4])
        return DataLoader(test_data, batch_size=self.eval_batch_size)



class EdgeModel(torch.nn.Module):
    def __init__(self, edge_in_dim, hidden_dim, edge_out_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP(edge_in_dim, hidden_dim, edge_out_dim,
                       2, 0.3, batch_norm=False, layer_norm=True)

    def forward(self, src, dest, edge_attr, u,batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim_1, hidden_dim_2):
        super(NodeModel, self).__init__()
        mlp_1_in_dim = node_in_dim + edge_in_dim
        mlp_2_in_dim = 256 + node_in_dim
        self.node_mlp_1 = MLP(mlp_1_in_dim, hidden_dim_1, 256,
                       2, 0.3, batch_norm=False, layer_norm=True)
        self.node_mlp_2 = MLP(mlp_2_in_dim, hidden_dim_2, 128,
                       2, 0.3, batch_norm=False, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u,batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, node_dim, global_in_dim, hidden_dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = MLP(node_dim + global_in_dim, hidden_dim, 128,
                       2, 0.3, batch_norm=False, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


#data.x = c_ids, data.edge_index = edge_index, data.edge_attr = rel_ids

class GraphNet(nn.Module):
    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0):
        super().__init__()
        self.init_range = init_range
        self.relation_num = relation_num
        self.ablation = ablation
        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)

        encoder_dim =  concept_dim * 2 + relation_dim
        edge_in_dim = encoder_dim
        self.EdgeModel = EdgeModel(edge_in_dim, 128, 128)
        self.NodeModel = NodeModel(concept_dim, 128, 128, 128)
        self.GlobalModel = None
        self.Network = MetaLayer(self.EdgeModel, self.NodeModel, self.GlobalModel)

    def forward(self, edge_index, c_ids, u, batch, num_tuples, rel_ids, emb_data = None):
        """
        edge_index: tensor of shape (2, E) E -> no. of edges
        c_ids: tensor of shape (N, ) N-> no. of nodes
        u: tensor of shape(batch_size,)
        num_tuples: tensor of shape (batch_size,)
        rel_ids: tensor of shape (E,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """
        #bs, _ = u.size()
        node_attr = self.concept_emb(c_ids, emb_data)
        edge_attr = self.rel_emb(rel_ids)
        x, edge_attr, u = self.Network(node_attr, edge_index, edge_attr, u,batch)
        return x, edge_attr, u


class Decoder(nn.Module):

    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0):

        super().__init__()
        self.GraphNet = GraphNet(concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                     hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout, pretrained_concept_emb=pretrained_concept_emb, pretrained_relation_emb=pretrained_relation_emb, freeze_ent_emb=freeze_ent_emb,
                     init_range=init_range, ablation=ablation, use_contextualized=use_contextualized, emb_scale=emb_scale)

        self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        self.activation = GELU()
        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout, batch_norm=False, layer_norm=True)

    def forward(self, sent_vecs, edge_index, c_ids, u, batch, num_tuples, rel_ids, max_tuple_num = 200, emb_data = None):
        #bs = .size()
        bs = num_tuples.shape[0]
        #print(bs)
        mask = torch.arange(max_tuple_num, device=c_ids.device) >= num_tuples.unsqueeze(1)
        mask[mask.all(1), 0] = 0
        node_vecs, edge_vecs, global_vecs = self.GraphNet(edge_index, c_ids, u, batch, num_tuples, rel_ids, emb_data)
        input = torch.empty(bs,max_tuple_num,128)
        qa_vecs =  torch.zeros_like(input, device = c_ids.device)
        j = 0
        for i in range(num_tuples.shape[0]):
            #print(num_tuples[i].item())
            num_tuples_1 = min(num_tuples[i],max_tuple_num)
            #print(node_vecs.shape)
            #print(node_vecs[j:j+num_tuples[i].item(),:].shape)
            #print(qa_vecs[i,:num_tuples_1,:].shape)
            #print(num_tuples[i])
            qa_vecs[i,:num_tuples_1,:] = node_vecs[j:j+num_tuples_1,:]
            j = j+num_tuples[i].item()
        qa_vecs = self.activation(qa_vecs)
        pooled_vecs, att_scores = self.attention(sent_vecs, qa_vecs, mask)
        logits = self.hid2out(self.dropout_m(torch.cat((pooled_vecs, sent_vecs), 1)))
        return logits, att_scores


class LMGraphNet(nn.Module):

    def __init__(self, model_name,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 num_attention_heads, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):

        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = Decoder(concept_num, concept_dim, relation_num, relation_dim, self.encoder.sent_dim, concept_in_dim,
                                   hidden_size, num_hidden_layers, num_attention_heads,
                                   fc_size, num_fc_layers, dropout, pretrained_concept_emb, pretrained_relation_emb,
                                   freeze_ent_emb=freeze_ent_emb, init_range=init_range, ablation=ablation,
                                   use_contextualized=use_contextualized, emb_scale=emb_scale)

    def forward(self, *inputs, layer_id, edge_index, c_ids, u, batch, rel_ids, num_tuples):
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        *lm_inputs ,= inputs
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs=sent_vecs, edge_index=edge_index, c_ids=c_ids, u = u,batch = batch, num_tuples=num_tuples, rel_ids=rel_ids)  # cxy-style param passing
        logits = logits.view(bs, nc)
        return logits, attn
# op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
# x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)

class LMGraphNetDataLoader(object):

    def __init__(self, train_statement_path, train_rpath_jsonl,
                 dev_statement_path, dev_rpath_jsonl,
                 test_statement_path, test_rpath_jsonl,
                 batch_size, eval_batch_size, device, model_name,
                 max_tuple_num=200, max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None, use_contextualized=False,
                 train_adj_path=None, train_node_features_path=None, dev_adj_path=None, dev_node_features_path=None,
                 test_adj_path=None, test_node_features_path=None, node_feature_type=None, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)

        num_choice = self.train_data[0].size(1)
        # self.train_data += load_2hop_relational_paths(train_rpath_jsonl, train_adj_path,
        #                                               emb_pk_path=train_node_features_path if use_contextualized else None,
        #                                               max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)[2]
        # self.dev_data += load_2hop_relational_paths(dev_rpath_jsonl, dev_adj_path,
        #                                             emb_pk_path=dev_node_features_path if use_contextualized else None,
        #                                             max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)[2]

        # #x = load_2hop_relational_paths(train_rpath_jsonl, train_adj_path,
        #                                               emb_pk_path=train_node_features_path if use_contextualized else None,
        #                                               max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)[2]
        #print(len(x))
        # self.train_graph = open_graph(train_adj_path)
        # self.dev_graph = open_graph(dev_adj_path)
        # self.test_graph = open_graph(test_adj_path)
        print([x.size(0) for x in [self.train_labels] + self.train_data])
        #print(k)
        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
            # self.test_data += load_2hop_relational_paths(test_rpath_jsonl, test_adj_path,
            #                                              emb_pk_path=test_node_features_path if use_contextualized else None,
            #                                              max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)[2]
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

        num_tuple_idx = -2 if use_contextualized else -1
        # print('| train_num_tuples = {:.2f} | dev_num_tuples = {:.2f} | test_num_tuples = {:.2f} |'.format(self.train_data[num_tuple_idx].float().mean(),
        #                                                                                                   self.dev_data[num_tuple_idx].float().mean(),
        #                                                                                                   self.test_data[num_tuple_idx].float().mean() if test_statement_path else 0))

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

    def __getitem__(self, index):
        raise NotImplementedError()

    def get_node_feature_dim(self):
        return self.train_data[-1].size(-1) if self.use_contextualized else None

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def get_train_indexes(self):
        n_train = self.inhouse_train_indexes.size(0)
        train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        return train_indexes


    def get_test_indexes(self):
        return self.inhouse_test_indexes

    def train(self, index):
        if self.is_inhouse:
            #n_train = self.inhouse_train_indexes.size(0)
            train_indexes = index
            #print(train_indexes)
            #print(k)
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def train_eval(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self, index):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, index, self.train_qids, self.train_labels, tensors=self.train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
