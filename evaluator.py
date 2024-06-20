import torch
import logging

import numpy as np

from tqdm import tqdm


def evaluate(my_model, data, mode='valid', que_triplets="all", limited_candidate=None):
    
    with torch.no_grad():
        # 1. check mode
        if mode == 'valid':
            query_triplets = torch.tensor(data.ori_valid).cuda()
            allR = torch.arange(end=data.num_rel).cuda()
            allE2allHR_matrix = torch.FloatTensor(data.ori_allE2allHR_matrix).cuda()
            ent_embs, hr_embs = my_model(allE2allHR_matrix, allE2allHR_matrix, allR)
        elif mode == 'test':
            data.update()
            my_model.num_ent, my_model.num_rel = data.num_ent, data.num_rel
            if que_triplets is "all":
                query_triplets = torch.tensor(data.emg_que).cuda()
            elif que_triplets == 'bri_seenR':
                query_triplets = torch.tensor(data.bri_seenR).cuda()
            elif que_triplets == 'bri_unseenR':
                query_triplets = torch.tensor(data.bri_unseenR).cuda()
            elif que_triplets == 'enc_seenR':
                query_triplets = torch.tensor(data.enc_seenR).cuda()
            elif que_triplets == 'enc_unseenR':
                query_triplets = torch.tensor(data.enc_unseenR).cuda()
            seenR = torch.LongTensor(data.seenR).cuda()
            allE2seenHR_matrix = torch.FloatTensor(data.emg_allE2seenHR).cuda()
            allE2allHR_matrix = torch.FloatTensor(data.emg_allE2allHR).cuda()
            ent_embs, hr_embs = my_model(allE2seenHR_matrix, allE2allHR_matrix, seenR)
        # 2. calculate score and ranks
        head_ranks = []
        tail_ranks = []
        ranks = []
        for triplet in tqdm(query_triplets):
            # 3. get one query triplet
            h, r, t = triplet
            # 4. head corrupt
            head_corrupt = triplet.unsqueeze(dim=0).repeat(data.num_ent, 1)
            head_corrupt[:, 0] = torch.arange(end=data.num_ent)
            # 5. get head rank
            head_scores = my_model.score(head_corrupt, ent_embs, hr_embs)
            head_filters = data.filter_dic[('_', int(r), int(t))]
            head_rank = get_rank(triplet, head_scores, head_filters, limited_candidate, target=0)
            # 6. tail corrupt
            tail_corrupt = triplet.unsqueeze(dim=0).repeat(data.num_ent, 1)
            tail_corrupt[:, 2] = torch.arange(end=data.num_ent)

            tail_scores = my_model.score(tail_corrupt, ent_embs, hr_embs)
            tail_filters = data.filter_dic[(int(h), int(r), '_')]
            tail_rank = get_rank(triplet, tail_scores, tail_filters, limited_candidate, target=2)

            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)
            # print(head_rank, tail_rank)


        logging.info(f"=========={mode} {que_triplets} LP==========")
        a_mr, a_mrr, a_hit1, a_hit3, a_hit10 = get_metrics(ranks)
        h_mr, h_mrr, h_hit1, h_hit3, h_hit10 = get_metrics(head_ranks)
        t_mr, t_mrr, t_hit1, t_hit3, t_hit10 = get_metrics(tail_ranks)
        logging.info(f"A | MR: {a_mr:.1f} | MRR: {a_mrr:.3f} | Hits@1: {a_hit1:.3f} | Hits@3: {a_hit3:.3f} | Hits@10: {a_hit10:.3f}")
        logging.info(f"H | MR: {h_mr:.1f} | MRR: {h_mrr:.3f} | Hits@1: {h_hit1:.3f} | Hits@3: {h_hit3:.3f} | Hits@10: {h_hit10:.3f}")
        logging.info(f"T | MR: {t_mr:.1f} | MRR: {t_mrr:.3f} | Hits@1: {t_hit1:.3f} | Hits@3: {t_hit3:.3f} | Hits@10: {t_hit10:.3f}")
        
    return a_mr, a_mrr, a_hit10, a_hit3, a_hit1

    
def get_rank(triplet, scores, filters, limited_candidate, target=0):
    thres = scores[triplet[target]].item()
    scores[filters] = thres - 1
    if limited_candidate is None:
        rank = (scores > thres).sum() + 1
    else:
        scores = np.random.choice(scores.cpu(), limited_candidate)
        rank = (scores > thres).sum() + 1
    return rank.item()


def get_metrics(rank):
	rank = np.array(rank, dtype = np.int)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit1, hit3, hit10