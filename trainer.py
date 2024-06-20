import os
import time
import torch
import logging

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from utils import sample_neg_triplets, move_batch_data_to_device
from evaluator import evaluate


class Trainer():
    
    def __init__(self, params, model, train_data):
        # initialization
        self.params = params
        self.model = model
        self.train_data = train_data
        # number of parameters
        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))
        # optimizer
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        # loss function
        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')
        # reset training
        self.reset_training_state()


    def reset_training_state(self):
        self.best_mr = 100000
        self.best_mrr = 0
        self.last_metric = 0
        self.not_improved_count = 0
    
    
    def train_epoch(self):
        # 0. variables
        total_loss = 0
        # 1. split train_data
        allE2seenR_matrix, allE2allR_matrix, seenR, query_triplets = self.train_data.split(0.9, 0.5)
        # 2. sample negative data
        pos_query_triplets, neg_query_triplets = sample_neg_triplets(query_triplets, self.train_data.num_ent, self.params.num_neg)
        # 3. prepare data and grad
        allE2seenR_matrix, allE2allR_matrix, seenR, pos_query_triplets, neg_query_triplets = move_batch_data_to_device(
            allE2seenR_matrix, allE2allR_matrix, seenR, pos_query_triplets, neg_query_triplets, self.params.device)
        self.optimizer.zero_grad()
        # 4. model forward
        ent_embs, hr_embs = self.model(allE2seenR_matrix, allE2allR_matrix, seenR)
        # 5. calculate pos scores
        pos_query_scores = self.model.score(pos_query_triplets, ent_embs, hr_embs)
        pos_query_scores = pos_query_scores.unsqueeze(dim=1).repeat(self.params.num_neg, 1).view(-1)
        # 6. calculate neg scores
        neg_query_scores = self.model.score(neg_query_triplets, ent_embs, hr_embs)
        # 7. calculate loss
        loss = self.criterion(pos_query_scores, neg_query_scores, 1*torch.ones(len(pos_query_scores)).to(self.params.device))
        # 8. backward
        loss.backward()
        self.optimizer.step()
        # 9. collect information
        with torch.no_grad():
            total_loss += loss
        # return
        return total_loss, ent_embs


    def train(self):
        # 0. reset training
        self.reset_training_state()
        # 1. train
        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            total_loss, ent_embs = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info('Epoch {:d} with loss: {:.3f}, best validation MRR: {:.3f}, in {:3f}'.format(epoch, total_loss, self.best_mrr, time_elapsed))
            # 2. valid
            if epoch % self.params.eval_every == 0:
                self.model.eval()
                mr, mrr, hit10, hit3, hit1 = evaluate(self.model, self.train_data)
                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    self.save_best_model()
                elif mrr == self.best_mrr and mr < self.best_mr:
                    self.best_mr = mr
                    self.save_best_model()
                
                self.save_latest_model() 
                self.model.train()
    
    
    def save_best_model(self):
        torch.save(self.model, os.path.join(self.params.exp_dir, 'best.pth'))
        logging.info('Better models found. Saved it!')
    
    
    def save_latest_model(self):
        torch.save(self.model, os.path.join(self.params.exp_dir, 'latest.pth'))