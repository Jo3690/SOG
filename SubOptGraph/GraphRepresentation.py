import os
import glob
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('path/envs/pytorch/share/RDKit/Data','BaseFeatures.fdef')  # 
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

def make_fc_edge_idx(node_num):
    fc_m = np.ones([node_num, node_num], dtype=np.int64)
    fc_coo = coo_matrix(fc_m)
    fc_edge_idx = np.array([fc_coo.row, fc_coo.col], dtype=np.int64)
    return fc_edge_idx



def atom_featurizer(mol):
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
    for idx in range(num_atoms):  #遍历mol中的每个原子
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
        atom_type.append(is_aromatic) #是否芳香性
        atom_type += hybridization # 杂化方式
        is_ring = int(atom.IsInRing())
        atom_type.append(is_ring)  # 是否是在环中
        atom_type += of_formal_charges

        node_feats.append(atom_type)  #原子特征
    return np.array(node_feats, dtype=np.float32), node_name


        #start_coor = [i for i in conf.GetAtomPosition(start)]
        #end_coor = [i for i in conf.GetAtomPosition(end)]
        #b_length = np.linalg.norm(np.array(end_coor)-np.array(start_coor))
        #b_type.insert(0, b_length)
def bond_featurizer(mol):
    #conf = mol.GetConformer()
    
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
    adjacency = adjacency.astype(np.float32)
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        bond_type = b.GetBondType().__str__()  # ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
        if bond_type == 'AROMATIC':
            adjacency[start,end] = 1.5
            adjacency[end,start] = 1.5
        elif bond_type == 'DOUBLE':
            adjacency[start,end] = 2
            adjacency[end,start] = 2
        elif bond_type == 'TRIPLE':
            adjacency[start,end] = 3
            adjacency[end,start] = 3
    return adjacency

class Mol2Graph(object):
    def __init__(self, mol, **kwargs):
        self.mol = mol
        self.x, self.node_name = atom_featurizer(mol)
        self.adjacency = bond_featurizer(mol)
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
