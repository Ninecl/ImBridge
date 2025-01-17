import os
import torch
import logging
import argparse

from tqdm import tqdm

from model.ImBridge import ImBridge
from trainer import Trainer
from dataset import GraphDataset


def main(params):
    
    data_path = os.path.join("./data", params.dataset)
    graph_dataset = GraphDataset(data_path)
    model = ImBridge(params, graph_dataset.entity2id, graph_dataset.relation2id).to(params.device)
    trainer = Trainer(params, model, graph_dataset)
    trainer.train()
    


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Imbridge model')
    
    # Experiment setup
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--disable_cuda", action='store_true',
                        help="Whether disable the gpu(s)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    
    # Model parameters setup
    parser.add_argument("--ent_dim", type=int, default=32,
                        help="The embedding dimension of entity")
    parser.add_argument("--rel_dim", type=int, default=64,
                        help="The embedding dimension of relation")
    parser.add_argument("--att_dim", type=int, default=256,
                        help="The dimension of attention in multi-head attention")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The dimension of hidden layer in ffn in multi-head attention")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="The number of attention head")
    parser.add_argument("--att_drop_prob", type=float, default=0.1,
                        help="The dropout prob in attention")
    parser.add_argument("--out_method", type=str, default='sum', choices=['sum', 'mean'],
                        help="The output method in multi-head attention")
    
    # Training setup
    parser.add_argument("--optimizer", "-opt", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=5,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--num_neg", type=int, default=16,
                        help="The number of negative samples for each positive link")
    parser.add_argument("--num_epochs", type=int, default=20000,
                        help="The number training epochs")
    parser.add_argument("--eval_every", type=int, default=200,
                        help="evaluate on valid dataset evey n epoch")
    
    params = parser.parse_args()
    
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    
    params.exp_dir = os.path.join("./checkpoint", f"{params.experiment_name}", f"{params.dataset}")
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)
    # log file
    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "train.log"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
        
    main(params)