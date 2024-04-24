import sys 
sys.path.append("..")  

import torch
from core.config import cfg,update_cfg
from core.train_helper import run 
from core.changed_model import GNNAsKernel
from core.transform import SubgraphsTransform
import random


from core.data import calculate_stats

import torch.nn.functional as F
import os
import os.path as osp
import pickle
import shutil
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
import argparse
from Mol2Graph import Mol2Graph
from core.model_utils.elements import cal_des
from torch_geometric.loader import DataLoader


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from transfer_package import create_model,TADF,test,train

path = 'path' #path used for the validation model parameter



def create_dataset(cfg): 
    

    transform_eval = SubgraphsTransform(cfg.subgraph.hops, 
                                        walk_length=cfg.subgraph.walk_length, 
                                        p=cfg.subgraph.walk_p, 
                                        q=cfg.subgraph.walk_q, 
                                        repeat=cfg.subgraph.walk_repeat,
                                        sampling_mode=None, 
                                        random_init=False)

    
    root = '/data/abs'  #data path 

    test_dataset = TADF(root, subset=0, split='val', file_name='test_list.txt',test=False,transform=transform_eval,percentage=0.0,mean=mean,std=std) 

    
    test_dataset = [x for x in test_dataset] 

    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=cfg.num_workers,follow_batch=['edge_attr'])

    return  test_loader


data = [line.strip().split('\t') for line in open("path/absorption.txt").readlines()]  

y = [float(i[2]) for i in data]
Y = np.array([i for i in y])
mean = Y.mean()
std = Y.std()
mean = float(mean)
std = float(std)




if __name__ == '__main__':

    cfg.merge_from_file('path/absorption.yaml')
    cfg = update_cfg(cfg,None)
    model = create_model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(path))
    model = model.to(cfg.device)
    test_loader = create_dataset(cfg)
    
    test_mae, test_output, y_test = test(test_loader, model, evaluator=None, device=cfg.device,std=std)

    test_output = np.asarray(test_output) 
    y_test = np.asarray(y_test)
    y_test = y_test*std + mean
    test_output = test_output*std + mean
    
    test_r2 = r2_score(y_test, test_output) 
    test_mae = mean_absolute_error(y_test,test_output)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_output))

    
    print('R2,MAE,RMSE',test_r2,test_mae,test_rmse)
