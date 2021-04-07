import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def train(args, data_helper, model, show_loss=True):

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    best_hit10 = 0.
    no_progress = 0
    for step in range(args.n_epoch):
        # training
        np.random.shuffle(data_helper.train_data)
        start = 0
        model.train()
        while start < data_helper.train_data.shape[0]:
            batch_data, batch_label = data_helper.get_next_batch(
                    data_helper.train_data, start, start+args.batch_size, args.device, args.n_negative)
            scores, regul = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(scores, batch_label)

            loss += args.weight_decay * regul

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / data_helper.train_data.shape[0] * 100, loss.item()))

        eval_result = evaluation(args, data_helper, model, test=False)
        print('epoch: {} h@10: {:.2f} t@10: {:.2f} a@10: {:.2f} hMRR: {:.2f} tMRR: {:.2f} aMRR: {:.2f}'.format(
                step,
                eval_result['head_hit10'], eval_result['tail_hit10'], eval_result['all_hit10'],
                eval_result['head_meanrank'], eval_result['tail_meanrank'], eval_result['all_meanrank']))
        if eval_result['all_hit10'] > best_hit10:
            best_hit10 = eval_result['all_hit10']
            torch.save(model.state_dict(), args.model_ckpt)
            print('saving model at epoch', step)
        else:
            no_progress += 1
        if no_progress > 2:
            break

def evaluation(args, data_helper, model, test=False):
    dataset = data_helper.test_data if test else data_helper.eval_data
    head_hit10 = 0.
    tail_hit10 = 0.
    head_meanrank = 0.
    tail_meanrank = 0.

    model.eval()
    for triple in tqdm(dataset):
        batch_head_eval, batch_tail_eval = data_helper.get_eval_batch(triple, args.device)
        head_scores, _ = model(batch_head_eval)
        tail_scores, _ = model(batch_tail_eval)
        head_rank = head_scores.argsort(descending=True)[0].item()
        tail_rank = tail_scores.argsort(descending=True)[0].item()
        if head_rank < 10:
            head_hit10 += 1
        if tail_rank < 10:
            tail_hit10 += 1
        head_meanrank += 1 / (head_rank + 1)
        tail_meanrank += 1 / (tail_rank + 1)

    head_hit10 /= len(dataset)
    tail_hit10 /= len(dataset)
    head_meanrank /= len(dataset)
    tail_meanrank /= len(dataset)
    return {'head_hit10': head_hit10 * 100.,
            'tail_hit10': tail_hit10 * 100.,
            'head_meanrank': head_meanrank,
            'tail_meanrank': tail_meanrank,
            'all_hit10': (head_hit10 + tail_hit10) * 100. / 2.,
            'all_meanrank': (head_meanrank + tail_meanrank) / 2.}
