import os
import time
import random
import logging

import numpy as np

from scipy.sparse import csc_matrix
from torch.utils.data import Dataset

from utils import readTriplets2Id, triplets2DglGraph, triplets2HyperRelation_matrix, hyperGraph2Matrix, get_rel_info


class GraphDataset(Dataset):
    
    def __init__(self, data_path) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.entity2id = dict()
        self.relation2id = dict()
        self.ori_train_path = os.path.join(data_path, "train.txt")
        self.ori_valid_path = os.path.join(data_path, "valid.txt")
        
        self.ori_train, self.entity2id, self.relation2id = readTriplets2Id(self.ori_train_path, 'hrt', self.entity2id, self.relation2id)
        self.ori_valid = readTriplets2Id(self.ori_valid_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.ori_all = self.ori_train + self.ori_valid
        
        self.num_ent = len(self.entity2id)
        self.num_rel = len(self.relation2id)
        self.rel_info = get_rel_info(self.ori_train)
        self.ori_train2idx = {triplet: idx for idx, triplet in enumerate(self.ori_train)}

        # dgl graph
        # self.g = triplets2DglGraph(self.ori_train, self.entity2id, self.relation2id)
        # hyper graph
        st = time.time()
        self.ori_allE2allHR_matrix = triplets2HyperRelation_matrix(self.ori_train, self.entity2id, self.relation2id)
        logging.info(f"Constructing relational hypergraph cost {time.time() - st}.")
        # fliter dic
        self.filter_dic = self.get_filter(self.ori_all)
        
        
    def split(self, support_percent, seenR_percent):
        support_triplet_idxs = []
        remained_triplets_idxs = np.ones(len(self.ori_train))
        for r, hts in self.rel_info.items():
            h, t = random.choice(hts)
            idx = self.ori_train2idx[(h, r, t)]
            remained_triplets_idxs[idx] = 0
            support_triplet_idxs.append(idx)
            
        num_support_triplet = int(len(self.ori_train) * support_percent)
        add_num = num_support_triplet - self.num_rel
        remained_nonzero_idxs = np.nonzero(remained_triplets_idxs)[0]
        np.random.shuffle(remained_nonzero_idxs)
        support_triplet_idxs = np.concatenate((np.array(support_triplet_idxs), remained_nonzero_idxs[: add_num]))
        query_triplet_idxs = remained_nonzero_idxs[add_num: ]     
        support_triplets = np.array(self.ori_train)[support_triplet_idxs]
        query_triplets = np.array(self.ori_train)[query_triplet_idxs]
        
        relations = np.array(list(self.relation2id.values()))
        num_seen_relations = int(len(relations) * seenR_percent + 0.5)
        np.random.shuffle(relations)
        seenR = np.sort(relations[: num_seen_relations])
        seenHR = np.concatenate((seenR, seenR+self.num_rel))

        head_hr_matrix = csc_matrix((np.ones(len(support_triplets)), (support_triplets[:, 0], support_triplets[:, 1])), shape=(self.num_ent, self.num_rel*2)).todense()
        tail_hr_matrix = csc_matrix((np.ones(len(support_triplets)), (support_triplets[:, 2], support_triplets[:, 1]+self.num_rel)), shape=(self.num_ent, self.num_rel*2)).todense()
        
        allE2allHR_matrix = head_hr_matrix + tail_hr_matrix
        allE2seenHR_matrix = allE2allHR_matrix[:, seenHR]

        return allE2seenHR_matrix, allE2allHR_matrix, seenR, query_triplets

    
    def update(self):
        emg_sup_path = os.path.join(self.data_path, 'support.txt')
        emg_que_path = os.path.join(self.data_path, 'query.txt')
        bri_seenR_path = os.path.join(self.data_path, 'query_bri_seenR.txt')
        bri_unseenR_path = os.path.join(self.data_path, 'query_bri_unseenR.txt')
        enc_seenR_path = os.path.join(self.data_path, 'query_enc_seenR.txt')
        enc_unseenR_path = os.path.join(self.data_path, 'query_enc_unseenR.txt')
        
        self.seenR = np.arange(self.num_rel)
        self.emg_sup, self.entity2id, self.relation2id = readTriplets2Id(emg_sup_path, 'hrt', self.entity2id, self.relation2id)
        self.emg_que = readTriplets2Id(emg_que_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.bri_seenR = readTriplets2Id(bri_seenR_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.bri_unseenR = readTriplets2Id(bri_unseenR_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.enc_seenR = readTriplets2Id(enc_seenR_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.enc_unseenR = readTriplets2Id(enc_unseenR_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        
        self.emg_all = self.emg_sup + self.ori_train
        self.num_rel = len(self.relation2id)
        self.num_ent = len(self.entity2id)
        
        seenHR = np.concatenate((self.seenR, self.seenR + self.num_rel))
        self.emg_allE2allHR = triplets2HyperRelation_matrix(self.emg_all, self.entity2id, self.relation2id)
        self.emg_allE2seenHR = self.emg_allE2allHR[:, seenHR]
        
        all_triplets = self.ori_train + self.ori_valid + self.emg_sup + self.emg_que
        self.filter_dic = self.get_filter(all_triplets)
    
    
    def get_filter(self, triplets):
        fliter_dic = {}
        
        for triplet in triplets:
            h, r, t = triplet
            if (h, r, '_') not in fliter_dic:
                fliter_dic[(h, r, '_')] = [t, ]
            else:
                fliter_dic[(h, r, '_')].append(t)
            if (h, '_', t) not in fliter_dic:
                fliter_dic[(h, '_', t)] = [r, ]
            else:
                fliter_dic[(h, '_', t)].append(r)
            if ('_', r, t) not in fliter_dic:
                fliter_dic[('_', r, t)] = [h, ]
            else:
                fliter_dic[('_', r, t)].append(h)
                
        return fliter_dic