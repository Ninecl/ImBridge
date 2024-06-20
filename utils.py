import os
import dgl
import torch

import numpy as np


def readTriplets2Id(path, mode, entity2id, relation2id, with_head=False, allow_emerging=True):

    triplets = []
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        
        ent_cnt = len(entity2id)
        rel_cnt = len(relation2id)
        
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            
            if allow_emerging:
                if h not in entity2id:
                    entity2id[h] = ent_cnt
                    ent_cnt += 1
                if t not in entity2id:
                    entity2id[t] = ent_cnt
                    ent_cnt += 1
                if r not in relation2id:
                    relation2id[r] = rel_cnt
                    rel_cnt += 1
            
            triplets.append((entity2id[h], relation2id[r], entity2id[t]))
    
    if allow_emerging:
        return triplets, entity2id, relation2id
    else:
        return triplets


def triplets2DglGraph(triplets, entity2id, relation2id):
    triplets = torch.LongTensor(triplets)
    g = dgl.graph((triplets[:, 0], triplets[:, 2]), num_nodes=len(entity2id))
    g.edata['type'] = triplets[:, 1]
    return g


def triplets2HyperRelation_matrix(triplets, entity2id, relation2id):
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    hr_matrix = np.zeros((num_entity, num_relation*2))
    
    for h, r, t in triplets:
        hr_matrix[h][r] += 1
        hr_matrix[t][r+num_relation] += 1
    
    return np.array(hr_matrix)


def hyperGraph2Matrix(hg):
    pass


def sample_neg_triplets(triplets, num_ent, num_neg):
    neg_triplets = torch.LongTensor(triplets).unsqueeze(dim=1).repeat(1, num_neg, 1)
    rand_result = torch.rand((len(triplets), num_neg))
    perturb_head = rand_result < 0.5
    perturb_tail = rand_result >= 0.5
    rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets), num_neg))
    neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
    neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
    return torch.LongTensor(triplets), neg_triplets


def get_rel_info(triplets):
    dic = {}
    for h, r, t in triplets:
        if r not in dic:
            dic[r] = [[h, t], ]
        else:
            dic[r].append([h, t])
    return dic


def move_batch_data_to_device(allE2seenR_matrix, allE2allR_matrix, seenR, pos_query_triplets, neg_query_triplets, device):
    allE2seenR_matrix = torch.FloatTensor(allE2seenR_matrix).to(device)
    allE2allR_matrix = torch.FloatTensor(allE2allR_matrix).to(device)
    seenR = torch.LongTensor(seenR).to(device)
    pos_query_triplets = torch.LongTensor(pos_query_triplets).to(device)
    neg_query_triplets = torch.LongTensor(neg_query_triplets).view(-1, 3).to(device)
    
    return allE2seenR_matrix, allE2allR_matrix, seenR, pos_query_triplets, neg_query_triplets


def move_test_data_to_device(emg_allE2allHR, emg_allE2seenHR, seenR, device):
    emg_allE2allHR = torch.FloatTensor(emg_allE2allHR).to(device)
    emg_allE2seenHR = torch.FloatTensor(emg_allE2seenHR).to(device)
    seenR = torch.LongTensor(seenR).to(device)
    
    return emg_allE2allHR, emg_allE2seenHR, seenR