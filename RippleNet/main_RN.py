import argparse
import numpy as np
from data_loader import load_data
from train_RN import train, eval_acc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def main2(model = None):
    np.random.seed(555)
    tf.reset_default_graph()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
    parser.add_argument('--ratio', type=float, default=0.4, help='size of training dataset')
    parser.add_argument('--type', type=str, default='edge1', help='size of training dataset')
    parser.add_argument('--num_pert', type=int, default=499470, help='size of training dataset')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')

    # parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    # parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    # parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    # parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    # parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
    # parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
    # parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    # parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
    # parser.add_argument('--type', type=str, default='rel', help='size of training dataset')
    # parser.add_argument('--num_pert', type=int, default=0, help='size of training dataset')
    # parser.add_argument('--item_update_mode', type=str, default='plus_transform',
    #                     help='how to update item at the end of each hop')
    # parser.add_argument('--using_all_hops', type=bool, default=True,
    #                     help='whether using outputs of all hops or just the last hop when making prediction')

    '''
    # default settings for Book-Crossing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    '''

    args = parser.parse_args('')

    show_loss = False
    data_info = load_data(args, type = args.type, num_pert = args.num_pert)
    return (eval_acc(args, data_info, show_loss))

if __name__ == '__main__':
    main2()

