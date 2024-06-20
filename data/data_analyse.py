import os
import numpy as np


SEEN_RATIO = 0.6
UNSEEN_R_RATIO = 0.5


def read2IdFile(path):
    dic = dict()
    with open (path, 'r') as f:
        
        for line in f.readlines():
            k, v = line.strip().split()
            dic[k] = int(v)
    return dic


def read_triplets2id(path, mode, entity2id=dict(), relation2id = dict(), with_head=False):
    triplets = []
    
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            
            triplets.append([entity2id[h], relation2id[r], entity2id[t]])
    
    return triplets


def data_split(triplets, ratio):
    e_degree_dic = {}
    r_degree_dic = {}
    self_loop_e = set()
    for h, r, t in triplets:
        # h
        if h in e_degree_dic:
            e_degree_dic[h] += 1
        else:
            e_degree_dic[h] = 1
        # t
        if t in e_degree_dic:
            e_degree_dic[t] += 1
        else:
            e_degree_dic[t] = 1
        # r
        if r in r_degree_dic:
            r_degree_dic[r] += 1
        else:
            r_degree_dic[r] = 1
        # self_loop
        if h == t:
            self_loop_e.add(h)
    
    selected_triplets_idxs = set()
    remained_triplets_idxs = list()
    
    # 删除孤点
    for i in range(0, len(triplets)):
        h, r, t = triplets[i]
        if h == t and e_degree_dic[h] == 2:
            e_degree_dic[h] -= 2
            r_degree_dic[r] -= 1
        else:
            remained_triplets_idxs.append(i)
    
    num_selected_triplets = int(len(remained_triplets_idxs) * ratio)
    cnt = 0
    while cnt < num_selected_triplets:
        idx = np.random.choice(remained_triplets_idxs)
        h, r, t = triplets[idx]
        if h == t:
            continue
        elif h not in self_loop_e and t not in self_loop_e and \
            e_degree_dic[h] - 1 > 0 and e_degree_dic[t] - 1 > 0 and r_degree_dic[r] - 1 > 0:
            pass
        elif h in self_loop_e and t not in self_loop_e and \
            e_degree_dic[h] - 1 > 2 and e_degree_dic[t] - 1 > 0 and r_degree_dic[r] - 1 > 0:
            pass
        elif h not in self_loop_e and t in self_loop_e and \
            e_degree_dic[h] - 1 > 0 and e_degree_dic[t] - 1 > 2 and r_degree_dic[r] - 1 > 0:
            pass
            cnt += 1
        elif h in self_loop_e and t in self_loop_e and \
            e_degree_dic[h] - 1 > 2 and e_degree_dic[t] - 1 > 2 and r_degree_dic[r] - 1 > 0:
            pass
        else:
            continue
        
        selected_triplets_idxs.add(idx)
        remained_triplets_idxs = np.array(list(set(remained_triplets_idxs) - selected_triplets_idxs))
        e_degree_dic[h] -= 1
        e_degree_dic[t] -= 1
        r_degree_dic[r] -= 1
        cnt += 1
    
    remained_triplets = []
    selected_triplets = []
    for i in remained_triplets_idxs:
        remained_triplets.append(triplets[i])
    for i in selected_triplets_idxs:
        selected_triplets.append(triplets[i])
    
    return remained_triplets, selected_triplets
    

if __name__ == "__main__":
    
    DATANAME = 'FB15k-237'
    
    train_path = f"./{DATANAME}/train.txt"
    valid_path = f"./{DATANAME}/valid.txt"
    test_path = f"./{DATANAME}/test.txt"
    
    entity2id = read2IdFile(f"./{DATANAME}/entity2id.txt")
    relation2id = read2IdFile(f"./{DATANAME}/relation2id.txt")
    
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    train_triplets = read_triplets2id(train_path, 'hrt', entity2id, relation2id)
    valid_triplets = read_triplets2id(valid_path, 'hrt', entity2id, relation2id)
    test_triplets = read_triplets2id(test_path, 'hrt', entity2id, relation2id)
    all_triplets = train_triplets + valid_triplets + test_triplets
    
    degree_count = np.zeros([len(entity2id), ])
    for triplet in all_triplets:
        h, r, t = triplet
        degree_count[h] += 1
        degree_count[t] += 1
    
    selected_entities = []
    for i in range(0, len(degree_count)):
        if degree_count[i] >= 15:
            selected_entities.append(i)
    
    # 确认seen的entity和relation
    num_seen_entity = int(len(selected_entities) * SEEN_RATIO)
    num_seen_relation = int(len(relation2id) * SEEN_RATIO)
    
    print("We sampled {} entities and {} relations as seen ones.".format(num_seen_entity, num_seen_relation))
    
    seen_entities = set(np.random.choice(selected_entities, num_seen_entity, False))
    seen_relations = set(np.random.choice(list(relation2id.values()), num_seen_relation, False))
    
    # 剩余全部entity都作为unseen entity，但是只保留unseen entity之间的unseen relation作为数据集中的新关系
    unseen_entities = set(selected_entities) - seen_entities
    
    # 确认enclosing中的unseen relation和enclosing link
    # 需要特别注意，这里并不是余下的所有relation都是unseen relation，而是在unseen entities之间的新关系才是unseen relation
    unseen_relations = set()
    for h, r, t in all_triplets:
        if h in unseen_entities and t in unseen_entities and r not in seen_relations:
            unseen_relations.add(r)
    print("The rest {} entities are taken as unseen entities.".format(len(unseen_entities)))
    print("There are {} relations remained between these unseen entities.".format(len(unseen_relations)))
    unseen_relations = list(unseen_relations)
    np.random.shuffle(unseen_relations)
    unseen_relations = set(unseen_relations[: int(len(unseen_relations) * UNSEEN_R_RATIO)])
    print("And we choose {}%, i.e., {} relations as unseen relations in the emerging KG.".format(int(UNSEEN_R_RATIO * 100), len(unseen_relations)))
        
    # 通过unseen relation再确认bridging的seen和unseen relation的link
    ori_seenR_triplets = []
    ori_unseenR_triplets = []
    enc_seenR_triplets = []
    enc_unseenR_triplets = []
    
    for h, r, t in all_triplets:
        if r in seen_relations and h in seen_entities and t in seen_entities:
            ori_seenR_triplets.append([h, r, t])
        elif r in unseen_relations and h in seen_entities and t in seen_entities:
            ori_unseenR_triplets.append([h, r, t])
        elif r in seen_relations and h in unseen_entities and t in unseen_entities:
            enc_seenR_triplets.append([h, r, t])
        elif r in unseen_relations and h in unseen_entities and t in unseen_entities:
            enc_unseenR_triplets.append([h, r, t])
            
    print("ori_seenR_triplets: ", len(ori_seenR_triplets))
    print("ori_unseenR_triplets: ", len(ori_unseenR_triplets))
    print("enc_seenR_triplets: ", len(enc_seenR_triplets))
    print("enc_unseenR_triplets: ", len(enc_unseenR_triplets))
    
    
    # 筛选数据
    # ori_seenR_triplets切分为训练时的训练集和测试集，按9:1的比例切分。
    print("Spliting data for ori_train and ori_valid...")
    ori_train, ori_valid = data_split(ori_seenR_triplets, 0.1)
        
    # emg部分由seenR和unseenR两部分构成, 需要分成support和query。
    # 先处理enclosing部分。
    print("Spliting data for emg_support_enc and emg_query_enc...")
    emg_support_enc_seenR, emg_query_enc_seenR = data_split(enc_seenR_triplets, 0.1)
    emg_support_enc_unseenR, emg_query_enc_unseenR = data_split(enc_unseenR_triplets, 0.1)
    emg_support = emg_support_enc_seenR + emg_support_enc_unseenR
    emg_query = emg_query_enc_seenR + emg_query_enc_unseenR
    
    # 重新确认ori和emg的entity set，ori entityset和emg entity set与seen entityset和unseen entityset并不相同
    ori_entity_set = set(np.array(ori_train)[:, 0]).union(set(np.array(ori_train)[:, 2]))
    emg_entity_set = set(np.array(emg_support)[:, 0]).union(set(np.array(emg_support)[:, 2]))
    # 统计bridging links
    bri_seenR_triplets = []
    bri_unseenR_triplets = []
    for h, r, t in all_triplets:
        if r in seen_relations and ((h in ori_entity_set and t in emg_entity_set) or (h in emg_entity_set and t in ori_entity_set)):
            bri_seenR_triplets.append([h, r, t])
        elif r in unseen_relations and ((h in ori_entity_set and t in emg_entity_set) or (h in emg_entity_set and t in ori_entity_set)):
            bri_unseenR_triplets.append([h, r, t])

    # bridging links里只取query的部分，且与enclosing部分数量相同
    emg_query_bri_seenR = []
    emg_query_bri_unseenR = []
    idxs = np.arange(len(bri_seenR_triplets))
    np.random.shuffle(idxs)
    for i in range(0, len(emg_query_enc_seenR)):
        emg_query.append(bri_seenR_triplets[idxs[i]])
        emg_query_bri_seenR.append(bri_seenR_triplets[idxs[i]])
    idxs = np.arange(len(bri_unseenR_triplets))
    np.random.shuffle(idxs)
    for i in range(0, len(emg_query_enc_unseenR)):
        emg_query.append(bri_unseenR_triplets[idxs[i]])
        emg_query_bri_unseenR.append(bri_unseenR_triplets[idxs[i]])
    
    
    # 写数据
    print("============SAVING DATA============")
    if DATANAME == "FB15k-237":
        STOREPATH = "FB-U{}".format(int(UNSEEN_R_RATIO * 100))
    print("We sampled {} triplets as train and {} triplets as valid in the original KG.".format(len(ori_train), len(ori_valid)))
    if not os.path.exists(STOREPATH):
        os.mkdir(STOREPATH)
    with open(f"./{STOREPATH}/train.txt", 'w') as f:
        for h, r, t in ori_train:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    with open(f"./{STOREPATH}/valid.txt", 'w') as f:
        for h, r, t in ori_valid:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    print("Sampled {} triplets as support and {} triplets as query in the emerging KG.".format(len(emg_support), len(emg_query)))
    with open(f"./{STOREPATH}/support.txt", 'w') as f:
        for h, r, t in emg_support:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    with open(f"./{STOREPATH}/query.txt", 'w') as f:
        for h, r, t in emg_query:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    print("In the query triplets, there are:")
    print("{} Enc_seenR".format(len(emg_query_enc_seenR)))
    with open(f"./{STOREPATH}/query_enc_seenR.txt", 'w') as f:
        for h, r, t in emg_query_enc_seenR:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    print("{} Enc_unseenR".format(len(emg_query_enc_unseenR)))      
    with open(f"./{STOREPATH}/query_enc_unseenR.txt", 'w') as f:
        for h, r, t in emg_query_enc_unseenR:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    print("{} Bri_seenR".format(len(emg_query_bri_seenR))) 
    with open(f"./{STOREPATH}/query_bri_seenR.txt", 'w') as f:
        for h, r, t in emg_query_bri_seenR:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
    print("{} Bri_unseenR".format(len(emg_query_bri_unseenR))) 
    with open(f"./{STOREPATH}/query_bri_unseenR.txt", 'w') as f:
        for h, r, t in emg_query_bri_unseenR:
            f.write("{}\t{}\t{}\n".format(id2entity[h], id2relation[r], id2entity[t]))
