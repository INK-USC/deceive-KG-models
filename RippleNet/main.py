import os
import argparse
import numpy as np
import torch

from utils import Data_Helper 
from train import train, evaluation
from model import ComplEx, DistMult

# for REPRODUCIBILITY
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--dim', type=int, default=50, help='dimension of entity and relation embeddings')
parser.add_argument('--n_negative', type=int, default=1, help='number of negative sampling per positive')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--gpu', type=int, default=0, help='gpu device')

args = parser.parse_args()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

data_helper = Data_Helper(args.dataset)

args.model_ckpt = os.path.join(args.save_dir, 'model.ckpt')
model = ComplEx(args.dim, data_helper.n_entity, data_helper.n_relation)
model.to(args.device)

if not os.path.exists(args.model_ckpt):
    train(args, data_helper, model)

model.load_state_dict(torch.load(args.model_ckpt, map_location=args.device))
eval_result = evaluation(args, data_helper, model, test=True)
print('Test result h@10: {:.2f} t@10: {:.2f} a@10: {:.2f} hMRR: {:.2f} tMRR: {:.2f} aMRR: {:.2f}'.format(
        eval_result['head_hit10'], eval_result['tail_hit10'], eval_result['all_hit10'],
        eval_result['head_meanrank'], eval_result['tail_meanrank'], eval_result['all_meanrank']))

