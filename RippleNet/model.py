import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplEx(nn.Module):
    def __init__(self, emb_dim, n_entity, n_relation):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.ent_re_embeddings = nn.Embedding(
            self.n_entity, self.emb_dim
        )
        self.ent_im_embeddings = nn.Embedding(
            self.n_entity, self.emb_dim
        )
        self.rel_re_embeddings = nn.Embedding(
            self.n_relation, self.emb_dim
        )
        self.rel_im_embeddings = nn.Embedding(
            self.n_relation, self.emb_dim
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):

        return -torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1,
        )

    def forward(self, batch_input):
        batch_h = batch_input[:, 0]
        batch_t = batch_input[:, 2]
        batch_r = batch_input[:, 1]

        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(h_im ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(t_im ** 2)
            + torch.mean(r_re ** 2)
            + torch.mean(r_im ** 2)
        )
        return score, regul

class DistMult(nn.Module):
    def __init__(self, emb_dim, n_entity, n_relation):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.ent_embeddings = nn.Embedding(
            self.n_entity, self.emb_dim
        )
        self.rel_embeddings = nn.Embedding(
            self.n_relation, self.emb_dim
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)

    def _calc(self, h, t, r):
        return - torch.sum(h * t * r, -1)

    def forward(self, batch_input):
        batch_h = batch_input[:, 0]
        batch_t = batch_input[:, 2]
        batch_r = batch_input[:, 1]

        h_re = self.ent_embeddings(batch_h)
        t_re = self.ent_embeddings(batch_t)
        r_re = self.rel_embeddings(batch_r)
        score = self._calc(h_re, t_re, r_re)
        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(r_re ** 2)
        )
        return score, regul