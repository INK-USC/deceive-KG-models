import random
from multiprocessing import cpu_count

from modeling.modeling_gn import *
from utils.optimization_utils import *
from utils.parser_utils import *
from utils.relpath_utils import *
from tqdm import tqdm
import os
import sys
import numpy as np
from Replay_Memory import *
import tensorflow as tf
from newprocess import *
import networkx as nx
from gn import *
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
from gn import *
from newprocess import *

# def load_model():
#     parser = get_parser()
#     args, _ = parser.parse_known_args('')
#     #parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
#     parser.add_argument('--save_dir', default=f'./saved_models/rn/rn/', help='model output directory')

#     # for finding relation paths
#     parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
#     parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
#     parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

#     # data
#     parser.add_argument('--train_rel_paths', default=f'./data/{args.dataset}/paths/train.relpath.2hop.jsonl')
#     parser.add_argument('--dev_rel_paths', default=f'./data/{args.dataset}/paths/dev.relpath.2hop.jsonl')
#     parser.add_argument('--test_rel_paths', default=f'./data/{args.dataset}/paths/test.relpath.2hop.jsonl')
#     parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
#     parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
#     parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
#     parser.add_argument('--train_node_features', default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--dev_node_features', default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--test_node_features', default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--train_concepts', default=f'./data/{args.dataset}/grounded/train.grounded.jsonl')
#     parser.add_argument('--dev_concepts', default=f'./data/{args.dataset}/grounded/dev.grounded.jsonl')
#     parser.add_argument('--test_concepts', default=f'./data/{args.dataset}/grounded/test.grounded.jsonl')

#     parser.add_argument('--node_feature_type', choices=['full', 'cls', 'mention'])
#     parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')
#     parser.add_argument('--max_tuple_num', default=200, type=int)

#     # model architecture
#     parser.add_argument('--ablation', default=['multihead_pool', 'None'], choices=['None', 'no_kg', 'no_2hop', 'no_1hop', 'no_qa', 'no_rel',
#                                                              'mrloss', 'fixrel', 'fakerel', 'no_factor_mul', 'no_2hop_qa',
#                                                              'randomrel', 'encode_qas', 'multihead_pool', 'att_pool'], nargs='+', const=None, help='run ablation test')
#     parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
#     parser.add_argument('--mlp_dim', default=128, type=int, help='number of MLP hidden units')
#     parser.add_argument('--mlp_layer_num', default=2, type=int, help='number of MLP layers')
#     parser.add_argument('--fc_dim', default=128, type=int, help='number of FC hidden units')
#     parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
#     parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
#     parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
#     parser.add_argument('--emb_scale', default=1.0, type=float, help='scale pretrained embeddings')

#     # regularization
#     parser.add_argument('--dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')

#     # optimization
#     parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
#     parser.add_argument('-mbs', '--mini_batch_size', default=4, type=int)
#     parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
#     parser.add_argument('--unfreeze_epoch', default=0, type=int)
#     parser.add_argument('--refreeze_epoch', default=10000, type=int)

#     parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
#     args = parser.parse_args('')
#     if args.debug:
#         parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)

#     # set ablation defaults
#     elif args.ablation == 'mrloss':
#         parser.set_defaults(loss='margin_rank')
#     args = parser.parse_args('')

#     model_path = os.path.join(args.save_dir, 'model_1.pt')

#     if 'lm' in args.ent_emb:
#         print('Using contextualized embeddings for concepts')
#         use_contextualized, cp_emb = True, None
#     else:
#         use_contextualized = False
#     cp_emb = [np.load(path) for path in args.ent_emb_paths]
#     cp_emb = torch.tensor(np.concatenate(cp_emb, 1))

#     concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

#     rel_emb = np.load(args.rel_emb_path)
#     rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
#     rel_emb = cal_2hop_rel_emb(rel_emb)
#     rel_emb = torch.tensor(rel_emb)
#     relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
#     #print('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
#     lstm_config = get_lstm_config_from_args(args)
#     print('model_loading')
#     #if model is None:
#     model = LMRelationNet(model_name=args.encoder, concept_num=concept_num, concept_dim=relation_dim,
#                           relation_num=relation_num, relation_dim=relation_dim,
#                           concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
#                           hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, num_attention_heads=args.att_head_num,
#                           fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
#                           pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb,
#                           init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized,
#                           emb_scale=args.emb_scale, encoder_config=lstm_config)
#     model.to(device)

#     model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
#     return model

def load_model():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    # parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred', 'decode'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/grn/', help='model output directory')

    # data
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--num_relation', default=34, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--train_embs', default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--dev_embs', default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--test_embs', default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')

    # model architecture
    parser.add_argument('-k', '--k', default=2, type=int, help='perform k-hop message passing at each layer')
    parser.add_argument('--ablation', default=[], choices=['no_trans', 'early_relu', 'no_att', 'ctx_trans', 'q2a_only',
                                                           'no_typed_transform', 'no_type_att', 'typed_pool', 'no_unary',
                                                           'detach_s_agg', 'detach_s_all', 'detach_s_pool', 'agg_self_loop',
                                                           'early_trans', 'pool_qc', 'pool_ac', 'pool_all',
                                                           'no_ent', 'no_rel', 'no_rel_att', 'no_1hop', 'fix_scale',
                                                           'no_lm'], nargs='*', help='run ablation test')
    parser.add_argument('-dd', '--diag_decompose', default=True, type=bool_flag, nargs='?', const=True, help='use diagonal decomposition')
    parser.add_argument('--num_basis', default=0, type=int, help='number of basis (0 to disable basis decomposition)')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--att_dim', default=50, type=int, help='dimensionality of the query vectors')
    parser.add_argument('--att_layer_num', default=1, type=int, help='number of hidden layers of the attention module')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--gnn_layer_num', default=1, type=int, help='number of GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--eps', type=float, default=1e-15, help='avoid numeric overflow')
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--init_rn', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--init_identity', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--fix_trans', default=False, type=bool_flag, nargs='?', const=True)

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.1, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args('')
    if args.simple:
        parser.set_defaults(diag_decompose=True, gnn_layer_num=1, k=1)
    # args = parser.parse_args('')

    print('Here')
    model_path = os.path.join(args.save_dir, 'model.pt')

    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized = True
    else:
        use_contextualized = False
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print('model_loading')

    lstm_config = get_lstm_config_from_args(args)
    model = LMGraphRelationNet(args.encoder, k=args.k, n_type=3, n_basis=args.num_basis, n_layer=args.gnn_layer_num,
                                diag_decompose=args.diag_decompose, n_concept=concept_num,
                                n_relation=args.num_relation, concept_dim=args.gnn_dim,
                                concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                                n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                att_dim=args.att_dim, att_layer_num=args.att_layer_num,
                                p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                ablation=args.ablation, init_range=args.init_range,
                                eps=args.eps, use_contextualized=use_contextualized,
                                do_init_rn=args.init_rn, do_init_identity=args.init_identity, encoder_config=lstm_config)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    return model

# def load_model():
#     parser = get_parser()
#     args, _ = parser.parse_known_args('')
#     # parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
#     parser.add_argument('--save_dir', default=f'./saved_models/gn/', help='model output directory')

#     # for finding relation paths
#     parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
#     parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
#     parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

#     # data
#     parser.add_argument('--train_rel_paths', default=f'./data/{args.dataset}/paths/train.relpath.2hop.jsonl')
#     parser.add_argument('--dev_rel_paths', default=f'./data/{args.dataset}/paths/dev.relpath.2hop.jsonl')
#     parser.add_argument('--test_rel_paths', default=f'./data/{args.dataset}/paths/test.relpath.2hop.jsonl')
#     parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
#     parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
#     parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
#     parser.add_argument('--train_node_features',
#                         default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--dev_node_features',
#                         default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--test_node_features',
#                         default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')
#     parser.add_argument('--train_concepts', default=f'./data/{args.dataset}/grounded/train.grounded.jsonl')
#     parser.add_argument('--dev_concepts', default=f'./data/{args.dataset}/grounded/dev.grounded.jsonl')
#     parser.add_argument('--test_concepts', default=f'./data/{args.dataset}/grounded/test.grounded.jsonl')

#     parser.add_argument('--node_feature_type', choices=['full', 'cls', 'mention'])
#     parser.add_argument('--use_cache', default=False, type=bool_flag, nargs='?', const=True,
#                         help='use cached data to accelerate data loading')
#     parser.add_argument('--max_tuple_num', default=200, type=int)

#     # model architecture
#     parser.add_argument('--ablation', default=None, choices=['None', 'no_kg', 'no_2hop', 'no_1hop', 'no_qa', 'no_rel',
#                                                              'mrloss', 'fixrel', 'fakerel', 'no_factor_mul',
#                                                              'no_2hop_qa',
#                                                              'randomrel', 'encode_qas', 'multihead_pool', 'att_pool'],
#                         nargs='?', const=None, help='run ablation test')
#     parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
#     parser.add_argument('--mlp_dim', default=128, type=int, help='number of MLP hidden units')
#     parser.add_argument('--mlp_layer_num', default=2, type=int, help='number of MLP layers')
#     parser.add_argument('--fc_dim', default=128, type=int, help='number of FC hidden units')
#     parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
#     parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
#                         help='freeze entity embedding layer')
#     parser.add_argument('--init_range', default=0.02, type=float,
#                         help='stddev when initializing with normal distribution')
#     parser.add_argument('--emb_scale', default=1.0, type=float, help='scale pretrained embeddings')

#     # regularization
#     parser.add_argument('--dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')

#     # optimization
#     parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
#     parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
#     parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
#     parser.add_argument('--unfreeze_epoch', default=0, type=int)
#     parser.add_argument('--refreeze_epoch', default=10000, type=int)

#     parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
#                         help='show this help message and exit')
#     args = parser.parse_args('')
#     if args.debug:
#         parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)

#     model_path = os.path.join(args.save_dir, 'model.pt')

#     if 'lm' in args.ent_emb:
#         print('Using contextualized embeddings for concepts')
#         use_contextualized, cp_emb = True, None
#     else:
#         use_contextualized = False
#     cp_emb = [np.load(path) for path in args.ent_emb_paths]
#     cp_emb = torch.tensor(np.concatenate(cp_emb, 1))

#     concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

#     rel_emb = np.load(args.rel_emb_path)
#     rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
#     # rel_emb = cal_2hop_rel_emb(rel_emb)
#     rel_emb = torch.tensor(rel_emb)
#     relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
#     print('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
#     lstm_config = get_lstm_config_from_args(args)
#     print('model_loading')

#     model = LMGraphNet(model_name=args.encoder, concept_num=concept_num, concept_dim=relation_dim,
#                         relation_num=relation_num, relation_dim=relation_dim,
#                         concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
#                         hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num,
#                         num_attention_heads=args.att_head_num,
#                         fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
#                         pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb,
#                         freeze_ent_emb=args.freeze_ent_emb,
#                         init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized,
#                         emb_scale=args.emb_scale, encoder_config=lstm_config)
#     model.to(device)

#     model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
#     return model


if __name__ == '__main__':
    main1()
    model = load_model()
    print(np.mean(main2(mode = 'eval', model = model)))
    with open('./data/cpnet/conceptnet_2.en.pruned.graph', 'rb') as handle:
        graph = pickle.load(handle)
    nx.write_gpickle(graph, './data/cpnet/conceptnet.en.pruned.graph')
    main1()
    print(np.mean(main2(mode = 'eval', model = model)))
