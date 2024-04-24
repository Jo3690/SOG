import sys 
sys.path.append("..")  
#sys.path.append("../..")  
import torch
from core.config import cfg,update_cfg
from core.train_helper import run_transfer 
from core.changed_model import GNNAsKernel
from core.transform import SubgraphsTransform
from core.data import calculate_stats
import numpy as np
from core.model_utils.elements import MLP
from torch import nn
from torch_geometric.loader import DataLoader
from transfer_package import create_model,TADF,test,train


emi = [line.strip().split('\t') for line in open('./data/plqy/raw/origin_plqy.txt').readlines()]
y = [float(i[2]) for i in emi]
Y = np.array([i for i in y])
mean = Y.mean()
std = Y.std()
mean=float(mean)
std=float(std)

def create_dataset(cfg): 
    # No need to do offline transformation
    transform = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True)

    transform_eval = SubgraphsTransform(cfg.subgraph.hops, 
                                        walk_length=cfg.subgraph.walk_length, 
                                        p=cfg.subgraph.walk_p, 
                                        q=cfg.subgraph.walk_q, 
                                        repeat=cfg.subgraph.walk_repeat,
                                        sampling_mode=None, 
                                        random_init=False)

    root = './train/data/bTADFqy'  
    train_dataset = TADF(root, subset=0, split='train', file_name='bTADFqy.txt',test=False,transform=transform,percentage=0.80,mean=mean,std=std) 
    val_dataset = TADF(root, subset=0, split='val', file_name='bTADFqy.txt',test=False,transform=transform_eval,percentage=0.80,mean=mean,std=std) 

    train_dataset = [x for x in train_dataset] 
    val_dataset = [x for x in val_dataset] 

    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
    val_loader = DataLoader(val_dataset,  cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers,follow_batch=['edge_attr'])



    return train_loader, val_loader, 


if __name__ == '__main__':


    
    path = 'checkpoint_path/ModelParams.pkl'
    cfg.merge_from_file('/home/xmpu215/215/emission/GNNAK/GNNAsKernel-main/train/configs/plqy.yaml')
    cfg = update_cfg(cfg,None)
    snapshot_path = cfg.model.path
    
    model = create_model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(path))
    for param in model.parameters():
            param.requires_grad_(False)


    model.output_decoder = nn.Sequential(MLP(181, 128, nlayer=2, with_final_activation=False),MLP(128, 1, nlayer=2, with_final_activation=False))
    model = model.to(cfg.device)

    run_transfer(cfg, create_dataset, model, train, test,snapshot_path=snapshot_path,mean=float(mean),std=float(std))
