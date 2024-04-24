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
    fdefName = os.path.join('/public/home/xmpu220/anaconda3/envs/pytorch/share/RDKit/Data','BaseFeatures.fdef')  # your own path
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
#possible_atom_type = ['H','Li', 'B', 'C', 'N', 'O',  'F', 'Na','Si', 'P', 'S', 'Cl', 'K','Ge', 'Se', 'Br', 'Sn','Te', 'I', 'Cs',] 
possible_atom_type =  ['Pb', 'Cl', 'S', 'Pt', 'Cd', 'V', 'I', 'Cu', 'Fe', 'Si', 'Ge', 'Ru', 'C', 'Pr', 'B', 'Sn', 'P', 'Zn', 'Hg', 'Ce', 'Hf', 'Ni', 'Br', 'Na', 'Ar', 'Mo', 'Se', 'Zr', 'F', 'N', 'H', 'O', 'Gd','Ti','W']
possible_hybridization = ['SP2', 'SP3', 'SP', 'S','SP3D','SP3D2', 'UNSPECIFIED']
possible_bond_type = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
PaulingElectroNegativity = {'H':2.20, 'Li':0.98, 'Be':1.57, 'B':2.04, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98, 'Ge':2.01, 'Se':2.55,
                            'Na':0.93, 'Mg':1.31, 'Al':1.61, 'Si':1.90, 'P':2.19, 'S':2.58, 'Cl':3.16, 'Ar':0.0,'Br':2.96, 'Sn':1.96,'Pb':1.87,'Cd':1.69,'V':1.63,'Cu':1.90,'Fe':1.83,'Ru':2.20,'Pr':1.13,'Zn':1.65,'Hg':2.00,'Ce':1.12,'Hf':1.30,
                            'Ni':1.91,'Mo':2.16,'Zr':1.33,'Gd':1.20,'Ti':1.54,'W':2.36,
                            'I':2.66, 'K':0.82, 'Ca':1.00, 'As':2.18, 'Te':2.10,'K':0.82,'Cs':0.79, }        # from https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)      

Vdw = {'H': 1.2, 'He': 1.4, 'Li': 2.2, 'Be': 1.9, 'B': 1.8, 'C': 1.7, 'N': 1.6, 'O': 1.55, 'F': 1.5,
       'Ne': 1.54, 'Na': 2.4, 'Mg': 2.2, 'Al': 2.1, 'Si': 2.1,'P': 1.95, 'S': 1.8, 'Cl': 1.8, 'Ar': 1.88,
       'K': 2.8, 'Ca': 2.4, 'Sc': 2.3, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05, 'Mn': 2.05, 'Fe': 2.05, 'Co': 2.0,
       'Ni': 2.0, 'Cu': 2.0, 'I': 2.1, 'Br': 1.9, 'Zn': 2.1, 'Ga': 2.1, 'Ge': 2.1, 'As': 2.05, 'Se': 1.9,
       'Kr': 2.02, 'Rb': 2.9, 'Sr': 2.55, 'Y': 2.4, 'Zr': 2.3, 'Nb': 2.15, 'Mo': 2.1, 'Tc': 2.05, 'Ru': 2.05,
       'Rh': 2.0, 'Pd': 2.05, 'Ag': 2.1, 'Cd': 2.2, 'In': 2.2, 'Sn': 2.25, 'Sb': 2.2, 'Te': 2.1,'Cs':3.0,'Pb':2.3,'Pt':2.05,'V':2.05,'Pd':2.05,'Pr':2.39,'Hg':2.05,'Ce':2.35,'Hf':2.25,'Ni':2.0,'Gd':2.37,'W':2.10,}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def GetChiral(atom):
    chiral_type = atom.GetChiralTag().__str__()
    is_chiral = []
    if chiral_type == 'CHI_UNSPECIFIED' or chiral_type == 'CHI_OTHER':
        is_chiral.append(0)
        is_chiral += [0, 0]
        return is_chiral
    else:
        is_chiral.append(1)
        is_chiral += one_of_k_encoding(
                                       chiral_type,
                                       ['CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW']
                                    )
        return is_chiral


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
        atomic_num = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        electro_negativity = PaulingElectroNegativity[symbol]
        is_aromatic = int(atom.GetIsAromatic())
        is_ring = int(atom.IsInRing())
        formal_charge=atom.GetFormalCharge()
        explicit_valence=atom.GetExplicitValence()
        implicit_valence=atom.GetImplicitValence()
        num_explicit_hs=atom.GetNumExplicitHs()
        num_radical_electrons=atom.GetNumRadicalElectrons()
        atom_type = one_of_k_encoding(symbol, possible_atom_type)   
        hybridization = one_of_k_encoding(atom.GetHybridization().__str__(), possible_hybridization)  
        atom_type.append(atomic_num)  
        
        atom_type.append(Vdw[symbol])  
        atom_type.append(is_donor[idx])  
        atom_type.append(is_acceptor[idx]) 
        atom_type.append(is_aromatic) 
        atom_type.append(electro_negativity)  
        atom_type += hybridization 
        atom_type.append(num_h)  
        atom_type.append(is_ring)  
        atom_type.append(formal_charge)
        atom_type.append(explicit_valence)
        atom_type.append(implicit_valence)
        atom_type.append(num_explicit_hs)
        atom_type.append(num_radical_electrons)
        node_feats.append(atom_type)  
    return np.array(node_feats, dtype=np.float32), node_name


        
        
        
def bond_featurizer(mol):
    
    n_atoms = mol.GetNumAtoms()
    bond_idx, bond_feats,atom_idx = [], [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)  
        is_inring = int(b.IsInRing())
        is_conjugated = int(b.GetIsConjugated())
        b_type.append(is_inring) 
        b_type.append(is_conjugated) 
        
        
        
        
        bond_idx.append([start, end])
        bond_idx.append([end, start])
        bond_feats.append(b_type)
        bond_feats.append(b_type)
        n_atom = list(range(n_atoms))
        
        for i in n_atom:
            e_ij = mol.GetBondBetweenAtoms(i,start)
            if e_ij is not None:
                atom_idx.append([i,start,start,end])
        n_atom = list(range(n_atoms))
        
        for j in n_atom:
            e_ij = mol.GetBondBetweenAtoms(j,end)
            if e_ij is not None:
                atom_idx.append([j,end,end,start])

    e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k:bond_idx[k])
    bond_idx = np.array(bond_idx)[e_sorted_idx]
    bond_feats = np.array(bond_feats, dtype=np.float32)[e_sorted_idx]
    a_sorted_idx = sorted(range(len(atom_idx)), key=lambda k:atom_idx[k])
    atom_idx = np.array(atom_idx)[a_sorted_idx]

    return bond_idx.astype(np.int64).T, bond_feats.astype(np.float32),atom_idx.astype(np.int64).T

class Mol2Graph(object):
    def __init__(self, mol,  **kwargs):
        self.mol = mol
        self.x, self.node_name = atom_featurizer(mol)
        self.edge_idx, self.edge_attr,self.atom_idx = bond_featurizer(mol)
        self.node_num = self.x.shape[0]
        
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
        