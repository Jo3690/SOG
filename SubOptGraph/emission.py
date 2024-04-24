import sys 
sys.path.append("..")  
#sys.path.append("../..")  
import torch
from core.config import cfg, update_cfg
from core.train_helper import run 
from core.changed_model import GNNAsKernel
from core.transform import SubgraphsTransform
import random

#from torch_geometric.datasets import ZINC
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

from Mol2Graph import Mol2Graph
from core.model_utils.elements import cal_des

class emi(object):
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__dict__[k] = kwargs[k] 


class Emitter(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        subset (boolean, optional): If set to :obj:`True`, will only load a
            subset of the dataset (12,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None,file_name=None,test=False,percentage=0.90):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        self.split = split
        self.file_name = file_name
        self.test = test
        self.percent = percentage
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        if self.file_name == None:
            return 'origin_emission.txt' 
        else:
            return self.file_name

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        if self.test == 1:
            return ['test.pt']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        f =  open(osp.join(self.raw_dir, self.raw_file_names), 'r').readlines()
        items = [line.strip().split('\t') for line in f]
        number = len(items)


        pbar = tqdm(total=number)
        pbar.set_description(f'Processing {self.split} dataset')
        data_list = []
        if os.path.exists(osp.join(self.raw_dir,f'data_list.pt')):
            data_list = torch.load(osp.join(self.raw_dir,f'data_list.pt'))

        else:
            Y = np.array([float(i[2]) for i in items])
            std = Y.std()
            mean = Y.mean()   
            for idx in range(number):
                if items[idx][1] == 'gas':
                    items[idx][1] = items[idx][0]
                mol1 = Chem.MolFromSmiles(items[idx][0])
                mol2 = Chem.MolFromSmiles(items[idx][1])        
                mol1 = Chem.AddHs(mol1)
                mol2 = Chem.AddHs(mol2)

                d = self.Graph(mol1,mol2)
    

                data = Data(x=torch.tensor(d.x,dtype=torch.float32),
                        edge_index=torch.tensor(d.edge_index.T, dtype=torch.int64),
                        edge_attr=torch.tensor(d.edge_feats,dtype=torch.int64),
                        y=torch.tensor((float(items[idx][2])-mean)/std, dtype=torch.float32),
                        global_state = torch.tensor(d.global_state, dtype=torch.float32),
                        subgraph_size = torch.tensor(d.subgraph_size,dtype = torch.int64),
                        v_shape = torch.tensor(d.v_shape,dtype = torch.int64),
                        atom_index = torch.tensor(d.atom_index.T,dtype=torch.int64),
                        e_idx = torch.tensor(d.e_idx,dtype=torch.int64).reshape(2,-1),
                        )
                
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                    #data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [data for data in data_list if self.pre_transform(data)]

                data_list.append(data)
                pbar.update(1)
            pbar.close()
            torch.save(data_list,osp.join(self.raw_dir,f'data_list.pt'))
       # for self.split in ['train', 'val', 'test']:
        if self.test == False:
            if self.split == 'train':
                data_list = data_list[:int(number*self.percent)]

            elif self.split == 'val':
                data_list = data_list[int(number*self.percent):]

            elif self.split == 'test':
                #if self.percent == 0.90:
                data_list = data_list[int(number*0.90):]

        else:
            data_list = data_list

        if self.subset:
            with open(osp.join(self.raw_dir, f'{self.split}.index'), 'r') as f:
                indices = [int(x) for x in f.read()[:-1].split(',')]
        #print(data_list)
        torch.save(self.collate(data_list),
                    osp.join(self.processed_dir, f'{self.split}.pt'))            


    def get_subgraph_size(self,mol1,mol2):
        atoms1 = mol1.GetNumAtoms()
        atoms2 = mol2.GetNumAtoms()
        return np.array([[atoms1,atoms2]])

    def Graph(self,mol1,mol2):
        g1 = Mol2Graph(mol1)
        g2 = Mol2Graph(mol2)
        x = np.concatenate([g1.x, g2.x], axis=0)
        edge_feats = np.concatenate([g1.edge_attr, g2.edge_attr], axis=0)
        e_idx2 = g2.edge_idx+g1.node_num
        atom_idx2 = g2.atom_idx+g1.node_num
        atom_index = np.concatenate([g1.atom_idx.T,atom_idx2.T],axis=0)  
        edge_index = np.concatenate([g1.edge_idx.T, e_idx2.T], axis=0)
        end_index = [] 
        start_index = []

        for i in atom_index:
            for idx,j in enumerate(edge_index):
                if (i[:2] == j).all():
                    end_index.append(idx)
                    break
            
            for idx,j in enumerate(edge_index):
                if (i[2:] == j).all():
                    start_index.append(idx)
                    break
        e_index = np.concatenate([np.array([end_index]),np.array([start_index])],axis=0)
        global_state_1 = cal_des(mol1).reshape(-1,5)
        global_state_2 = cal_des(mol2).reshape(-1,5)
        subgraph_size = self.get_subgraph_size(mol1,mol2)
        v_shape = np.array(x.shape[0])
        global_state = np.concatenate((global_state_1,global_state_2),axis = 0)

        return emi(x=x, edge_feats=edge_feats, edge_index=edge_index,global_state = global_state, subgraph_size = subgraph_size, v_shape = v_shape,atom_index=atom_index,e_idx=np.array(e_index,dtype=np.int64))

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

    root = './data/emission'  
    train_dataset = Emitter(root, subset=0, split='train', transform=transform,file_name='origin_emission.txt',percentage=0.80)
    val_dataset = Emitter(root, subset=0, split='val', transform=transform_eval,file_name='origin_emission.txt',percentage=0.80) 
    test_dataset = Emitter(root, subset=0, split='test', transform=transform_eval,file_name='origin_emission.txt',percentage=0.80)   

    train_dataset = [x for x in train_dataset] 
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 



    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = GNNAsKernel(nfeat_node=37, nfeat_edge=6,
                        nhid=cfg.model.hidden_size, 
                        nout=1, 
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=['OledConvne',cfg.model.gnn_type], 
                        hop_dim=cfg.model.hops_dim,
                        use_normal_gnn=cfg.model.use_normal_gnn, 
                        vn=cfg.model.virtual_node, 
                        pooling=cfg.model.pool,
                        embs=cfg.model.embs,
                        embs_combine_mode=cfg.model.embs_combine_mode,
                        mlp_layers=cfg.model.mlp_layers,
                        dropout=cfg.train.dropout, 
                        subsampling=True if cfg.sampling.mode is not None else False,
                        online=cfg.subgraph.online,
                        state_dim=cfg.model.state_dim) 
    return model

def train(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    
    #N = 0 
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        #with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = F.mse_loss(output, y)
        total_loss = loss.item() * num_graphs + total_loss
        #loss = (model(data).squeeze() - y).abs().mean()
        #
        loss.backward()
        #total_loss += loss.item() * num_graphs
        optimizer.step()
        #N += num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader, model, evaluator, device,std):
    model.eval()   
    error = 0
    loss_all = 0
    
    model_output = []
    y = []
    for data in loader:
        data = data.to(device)
        output = model(data).reshape(-1)

        error += (output * std - data.y * std).abs().sum().item()              
                                                                     
        loss = F.mse_loss(output, data.y)

        loss_all += loss.item() * data.num_graphs                           #   Returns a new Tensor, detached from the current graph. The result will never require gradient.
        model_output.extend(output.tolist())                                      
        y.extend(data.y.tolist())

    return loss_all, error/len(loader.dataset), model_output, y


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('/public/home/xmpu220/emission/GNNAK/GNNAK/train/configs/emission.yaml')
    cfg = update_cfg(cfg)
    emission = [line.strip().split('\t') for line in open('/public/home/xmpu220/emission/GNNAK/GNNAK/train/data/0.80emission/raw/origin_emission.txt').readlines()]
    y = [float(i[2]) for i in emission]
    Y = np.array([i for i in y])
    mean = Y.mean()
    std = Y.std()
    snapshot_path = cfg.model.path
    run(cfg, create_dataset, create_model, train, test,snapshot_path=snapshot_path,mean=float(mean),std=float(std))
