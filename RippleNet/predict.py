import os
import argparse
import numpy as np
import torch

from model import ComplEx

# for REPRODUCIBILITY
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device')

args = parser.parse_args('')

# args.device = torch.device('cuda'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
#
# model = torch.load('./checkpoints/movie_undirected_kbc.ckpt', map_location=args.device)

# input: [[head, relation, tail]]
# output: logit, regu_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./checkpoints/movie_undirected_kbc.ckpt', map_location=device)
model.to(device)
def pred(head,relation, tail):

    batch_data = torch.LongTensor([[head, relation, tail]]).to(device)
    scores, _ = model(batch_data)
    return(torch.sigmoid(scores).cpu())

if __name__=='__main__':
    # node_embs = np.zeros((102569, ))
    node_embs = None
    rel_embs = None

    for i in range(102569):
        if i%1000 == 0:
            print(i)
        x = torch.LongTensor([i]).to(device)
        embs = model.ent_embeddings(x)
        embs = embs.detach().cpu().numpy()
        if node_embs is None:
            node_embs = embs
        else:
            node_embs = np.vstack((node_embs, embs))

    np.save('node_embs.npy', node_embs)

    for i in range(32):
        x = torch.LongTensor([i]).to(device)
        embs = model.ent_embeddings(x)
        embs = embs.detach().cpu().numpy()
        if rel_embs is None:
            rel_embs = embs
        else:
            rel_embs = np.vstack((rel_embs, embs))

    np.save('rel_embs.npy', rel_embs)
