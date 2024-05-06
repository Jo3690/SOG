import os
import os.path as osp
import torch
from torch_geometric.data import Data,Dataset
from torch_geometric.data.batch import Batch
#import networkx as nx
import random
from collections import defaultdict
import numpy as np
#import xlrd
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
#from utils import Complete, angle, area_triangle, cal_dist

try:
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    from rdkit.Chem import AllChem
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
except:
    Chem, ChemicalFeatures, RDConfig, fdef_name, chem_feature_factory = 5 * [None]
    print('Please install rdkit for data processing')


PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('path/pytorch/share/RDKit/Data','BaseFeatures.fdef')  # 
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
#possible_atom_type = ['H', 'B', 'C', 'N', 'O',  'F', 'Na','Si', 'P', 'S', 'Cl','Ar', 'K','Ge', 'Se', 'Br', 'Sn','Te', 'I', 'Cs',] 
possible_atom_type = ['C', 'N','O','S','H','F','Na','Cl','Br','I','Se','Te','Si','P','B','Sn','Ge',] #
possible_hybridization = ['S','SP','SP2', 'SP3',  'SP3D','SP3D2','UNSPECIFIED'] # 'UNSPECIFIED'
of_formal_charge = [-4,-3,-2,-1,0,1,2,3,4] #
of_H = [0,1,2,3,4] #
of_atom = [0,1,2,3,4,5] #

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

class SpecifyTarget(object):
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        data.y = data.y[self.target].view(-1)
        return data
file_name_1 = []

def load_dataset(path,name='abs', percentage=0.80,transform=None):
    data_set = MyOpticalDataset(path, name,transform=transform)
    # shuffle data
    
    indices = list(range(len(data_set)))
    data_set = data_set[indices]    
    number = len(data_set)
    #one_tenth = len(data_set) // 5
    #train_dataset = data_set[:(number-one_tenth)]
    #test_dataset = data_set[: (number-one_tenth)]    
    
    train_dataset = data_set[:int(number*percentage)]
    test_dataset = data_set[int(number*percentage):]

    return train_dataset, test_dataset
    #return data_set,data_set
    
class MyDataset(Dataset):
    def atom_featurizer(self,mol):
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)
        feats = factory.GetFeaturesForMol(mol)
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                for u in feats[i].GetAtomIds():
                    is_donor[u] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                for u in feats[i].GetAtomIds():
                    is_acceptor[u] = 1
        num_atoms = mol.GetNumAtoms()
        node_feats = []
        node_name = []
        #node_idx = []
        for idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(idx)
            try:
                atom_name = atom.GetProp('_TriposAtomName')
            except:
                atom_name = atom.GetSymbol() + str(idx)
            node_name.append(atom_name)
        for idx in range(num_atoms):  
            atom = mol.GetAtomWithIdx(idx)
            
            symbol = atom.GetSymbol()
            num_h = atom.GetTotalNumHs()
            atoms_num = atom.GetTotalDegree()
            hybridization = one_of_k_encoding(atom.GetHybridization().__str__(), possible_hybridization)  #one-hot
            formal_charge=atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            
            atom_type = one_of_k_encoding(symbol, possible_atom_type)   #one-hot
            of_Hs = one_of_k_encoding(num_h,of_H)
            of_atoms = one_of_k_encoding(atoms_num,of_atom)
            of_formal_charges = one_of_k_encoding(formal_charge,of_formal_charge)
            atom_type += of_Hs
            atom_type += of_atoms
            atom_type.append(is_aromatic) #
            atom_type += hybridization # 
            is_ring = int(atom.IsInRing())
            atom_type.append(is_ring)  #
            atom_type += of_formal_charges

            node_feats.append(atom_type)  #
        return np.array(node_feats, dtype=np.float32)

    def bond_featurizer(self,mol):
        #conf = mol.GetConformer()
        bond_idx,edge_weight = [],[]
        
        for b in mol.GetBonds():
            start = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bond_type = b.GetBondType().__str__()  # ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
            if bond_type == 'AROMATIC':
                edge_weight.append(1.5)
                
            elif bond_type == 'DOUBLE':
                edge_weight.append(2)
            
            elif bond_type == 'SINGLE':
                edge_weight.append(1)
                
            elif bond_type == 'TRIPLE':
                edge_weight.append(3)
            bond_idx.append([start, end])
        #e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k:bond_idx[k])
        bond_idx = np.array(bond_idx)
        edge_weight = np.asarray(edge_weight)
        return bond_idx.astype(np.int64).T,edge_weight.astype(np.float32)

    def mol2graph(self, mol, y):
        if mol is None: return None
        # Build pyg data
        node_attr = self.atom_featurizer(mol)
        edge_index, edge_weight = self.bond_featurizer(mol)
        data = Data(
            x=torch.tensor(node_attr,dtype=torch.float32),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            edge_weight=torch.tensor(edge_weight,dtype=torch.float32),
            y=torch.tensor(y,dtype=torch.float32),  # None as a placeholder
            # name=mol.GetProp('_Name'),
        )
        return data
        
class MyOpticalDataset(MyDataset):
    def __init__(self, root, name,transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.data_list: List[Data]] = None
        #self.data_list_sol: List[Data]] = None  
        
        
        data = torch.load(osp.join(self.processed_dir,'data_{}.pt'.format(self.name)))
        data_sol = torch.load(osp.join(self.processed_dir,'data_sol_{}.pt'.format(self.name)))
        
        self.data_list = Batch.to_data_list(data)
        self.data_list_sol = Batch.to_data_list(data_sol)  
        

    @property
    def raw_file_names(self):
    
        return 'data.txt'

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        # 
        return ['data_{}.pt'.format(self.name),'data_sol_{}.pt'.format(self.name),]
        
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, list) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")
    
    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, list) else data
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")
    
    def download(self):
        pass
    
        
    def process(self):
        txt_path = self.root + "/data.txt" 
        items = [i.strip().split('\t') for i in open(txt_path).readlines()]

        data_list = []
        data_sol_list = []
        
        Y = np.array([float(i[2]) for i in items])
        std = Y.std()
        mean = Y.mean()   
        for i in range(len(items)):
            print("process:{}".format(i))
            smi = items[i][0]
            smi_sol = items[i][1]
            
            y = np.asarray(float(items[i][2]))
            y = (y-mean)/std
            
            mol = Chem.MolFromSmiles(smi)
            mol = AllChem.AddHs(mol)
            
            data = self.mol2graph(mol,y)
            data_list.append(data)
              
            mol_sol = Chem.MolFromSmiles(smi_sol)
            mol_sol = AllChem.AddHs(mol_sol)
                      
            data_sol = self.mol2graph(mol_sol,y)              
            data_sol_list.append(data_sol) 

        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            data_sol_list = [self.pre_transform(data) for data in data_sol_list]
        
    
        #data, slices = self.collate(data_list)
        loader_chro1 = Batch.from_data_list(data_list)
        loader_sol = Batch.from_data_list(data_sol_list)    
           
        torch.save(loader_chro1,osp.join(self.processed_dir,'data_{}.pt'.format(self.name)))
        torch.save(loader_sol,osp.join(self.processed_dir,'data_sol_{}.pt'.format(self.name)))
        
    def len(self):        
        dataloader_chro = torch.load(osp.join(self.processed_dir,'data_{}.pt'.format(self.name))) 
        return dataloader_chro.num_graphs
    
    def get(self,idx):
        return [self.data_list[idx],self.data_list_sol[idx]]
    
            
        
if __name__ == '__main__':
    
    load_dataset('data/abs','abs')
