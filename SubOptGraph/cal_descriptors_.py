# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdFreeSASA
import numpy as np
import math

def FractionNO(mol):
    return Descriptors.NOCount(mol) / float(mol.GetNumHeavyAtoms())

def FractionAromaticAtoms(mol):
    return len(mol.GetAromaticAtoms()) / float(mol.GetNumHeavyAtoms())

def NumHAcceptorsAndDonors(mol):
    return Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)

def RotatableBondNumber(mol):
    mol = Chem.RemoveHs(mol)  
    return Descriptors.NumRotatableBonds(mol) #/ float(mol.GetNumBonds())

def RingsNums(mol):
    alirings = Lipinski.NumAliphaticRings(mol)
    arorings = Lipinski.NumAromaticRings(mol)
    return alirings,arorings

def cal_des(mol):
    no = FractionNO(mol)
    ar = FractionAromaticAtoms(mol)
    rb = RotatableBondNumber(mol)
    al,ar = RingsNums(mol)
    return np.array([no,ar,rb,al,ar])
