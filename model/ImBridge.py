import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from model.attention import AttenEncoderLayer


class ImBridge(nn.Module):
    
    def __init__(self, params, entity2id, relation2id):
        super(ImBridge, self).__init__()
        
        self.params = params
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_ent = len(entity2id)
        self.num_rel = len(relation2id)
        
        self.ent_dim = params.ent_dim
        self.rel_dim = params.rel_dim
        self.att_dim = params.att_dim
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.att_drop_prob = params.att_drop_prob
        
        # hyper-relation embeddings
        self.hyper_head_embs = nn.Embedding(len(relation2id), self.rel_dim)
        self.hyper_tail_embs = nn.Embedding(len(relation2id), self.rel_dim)
        
        # encoder paramaters
        self.hra_1 = AttenEncoderLayer(self.rel_dim, self.att_dim, self.hidden_dim, self.num_heads, self.att_drop_prob)
        self.w_hr2e_1 = nn.Linear(self.rel_dim, self.ent_dim, bias=False)
        self.act_1 = nn.ReLU()
        self.w_e2hr_1 = nn.Linear(self.ent_dim, self.rel_dim, bias=False)
        self.hra_2 = AttenEncoderLayer(self.rel_dim, self.att_dim, self.hidden_dim, self.num_heads, self.att_drop_prob)
        self.w_hr2e_2 = nn.Linear(self.rel_dim, self.ent_dim, bias=False)
        self.act_2 = nn.Sigmoid()
        
        # decoder paramaters
        self.w_hr_score = nn.Linear(self.rel_dim*2, self.ent_dim, bias=False)
    
    def forward(self, allE2seenHR_matrix, allE2allHR_matrix, seenR):
        
        # 1. get embs of all seen hyper-relations
        seen_hh_embs = self.hyper_head_embs(seenR)
        seen_ht_embs = self.hyper_tail_embs(seenR)
        seen_hr_embs = torch.concatenate((seen_hh_embs, seen_ht_embs))
        # print('seen_hr_embs', seen_hr_embs)
        # 2. calculate self-attention between seen hyper-relations
        seen_hr_embs = self.hra_1(seen_hr_embs)
        # 3. get embeddings of all entities based on the seen hyper-relations
        allE2seenHR_matrix = F.normalize(allE2seenHR_matrix, p=1, dim=-1)
        all_ent_embs = self.act_1(allE2seenHR_matrix @ self.w_hr2e_1(seen_hr_embs))
        # print('all_ent_embs1', all_ent_embs)
        # 4. get embeddings of all hyper-relations based on all entities
        E = 1 / torch.sum(allE2allHR_matrix.transpose(0, 1), -1)
        # print('E', torch.sum(allE2allHR_matrix.transpose(0, 1), -1))
        all_hr_embs = self.w_e2hr_1(torch.diag(E) @ allE2allHR_matrix.transpose(0, 1) @ all_ent_embs)
        # 5. calculate self-attention between all hyper-relations
        all_hr_embs = self.hra_2(all_hr_embs)
        # print('all_hr_embs', all_hr_embs)
        # 6. get embeddings of all entities based on all hyper-relations
        allE2allHR_matrix = F.normalize(allE2allHR_matrix, p=1, dim=-1)
        all_ent_embs = self.act_2(allE2allHR_matrix @ self.w_hr2e_2(all_hr_embs))
        # print('all_ent_embs2', all_ent_embs)
        # input()
        
        return all_ent_embs, all_hr_embs
    
    
    def score(self, triplets, ent_embs, hr_embs):
        # 1. get idxs
        head_idxs = triplets[:, 0]
        hyper_head_idxs = triplets[:, 1]
        tail_idxs = triplets[:, 2]
        hyper_tail_idxs = triplets[:, 1] + self.num_rel
        # 2. get embs
        h_embs = ent_embs[head_idxs]
        hh_embs = hr_embs[hyper_head_idxs]
        t_embs = ent_embs[tail_idxs]
        ht_embs = hr_embs[hyper_tail_idxs]
        r_embs = self.w_hr_score(torch.concat((hh_embs, ht_embs), dim=-1))
        # 3. a * d - b * c
        # [head_embs, tail_embs]
        # [hh_embs, ht_embs]
        # scores = torch.sum(head_embs * ht_embs - tail_embs * hh_embs, dim=-1)
        scores = torch.sum(h_embs * r_embs * t_embs, dim=-1)
        
        return torch.abs(scores)
        