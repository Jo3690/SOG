import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import torch


from typing import Union, Optional
from numpy.core.multiarray import negative
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)  


from torch import Tensor

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor

from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot  
from torch_geometric.nn import MessagePassing,GlobalAttention, GraphNorm
from torch_scatter import scatter
from torch_geometric.utils import softmax


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
    return Descriptors.NumRotatableBonds(mol) 

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


BN = True

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=6, max_num_values=6): 
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0

        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) 
                                        or (not with_norm) else False) 
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) 

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = x.type(torch.float32)
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)  

        
        
        return x 

class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=BN):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm, with_final_activation=True, bias=not BN)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        G = global_add_pool(x, batch)
        if vn is not None:
            G += vn
        vn = self.mlp(G)
        x += vn[batch]
        return vn, x




class globalconv(torch.nn.Module):  
    def __init__(self,nhid,nout):
        super().__init__()
        self.lin1 = nn.Linear(nhid,64)
        self.lin2 = nn.Linear(64,32)
        self.lin3 = nn.Linear(32,nout)
        
        self.reset_parameters()  

    def reset_parameters(self):  
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        glorot(self.lin3.weight)
        
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        
        return x


class GlobalTransfer():
    
    def transfer(self,subgraph_size,global_state,v_shape):
        global_dim = global_state.shape[-1]
        t = torch.zeros(v_shape[0].item(),global_dim,device='cuda')
        
        for i in range(global_state.shape[0]):
            max_graph_size = v_shape[i]
            subg1 = torch.zeros(subgraph_size[i][0].item(),dtype = torch.int64,device='cuda')
            subg2 = torch.ones(subgraph_size[i][1].item(),dtype= torch.int64,device='cuda')
            cated = torch.cat([subg1,subg2],dim=0,)
            gather_tensor = global_state[i][cated]
            padding_num = torch.sub(max_graph_size.item(),torch.sum(subgraph_size[i]).item())
            pad = torch.nn.ZeroPad2d([0,0,0,padding_num.item()])
            padding = pad(gather_tensor)
            pad_reshape = padding.reshape(max_graph_size,global_dim,)
            t = torch.cat([t,pad_reshape],dim=0)
        return t[v_shape[0]:]



class EdgeConv(torch.nn.Module):  
    def __init__(self, in_channels, out_channels,node_dim=37,state_dim:int=5,aggr='add'):  
        super().__init__()  
        self.aggr = aggr
        self.lin_edge = nn.Linear(node_dim+in_channels+state_dim,out_channels)  
        self.lin_e = nn.Linear(in_channels,out_channels)
        
        self.reset_parameters()  

    def reset_parameters(self):  
        glorot(self.lin_edge.weight)
        glorot(self.lin_e.weight)
        
    def forward(self, x, edge_attr, atom_index,e_idx, global_state,):   
        
        x = torch.cat([x,global_state],dim=1) 

        edge_adj = torch.cat([edge_attr[e_idx[1]],x[atom_index[0]]],dim=-1)  
        
        edge_adj = F.elu(self.lin_edge(edge_adj)) 
        
        edge_out = scatter(edge_adj,e_idx[0],dim=0,reduce='add')
       
       
        
        edge_out = F.elu(self.lin_e(edge_attr)) + edge_out

        return edge_out

class NodeConv(torch.nn.Module):  
    def __init__(self, in_channels, out_channels,edge_dim=6,state_dim:int=5,aggr='add'):  
        super().__init__()  
        self.aggr = aggr
        self.lin_neg = nn.Linear(in_channels+edge_dim+state_dim, out_channels)   
        self.lin_root = nn.Linear(in_channels+state_dim, out_channels)  
    
        self.reset_parameters()  

    def reset_parameters(self):  
        glorot(self.lin_neg.weight)
        glorot(self.lin_root.weight)
        
    def forward(self, x, edge_index, edge_attr, global_state,):   

        x = torch.cat([x,global_state],dim=1) 

        x_adj = torch.cat([x[edge_index[1]],edge_attr], dim=1)   
        x_adj = F.elu(self.lin_neg(x_adj))

        neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr)   
       
       
       
        x_out = F.elu(self.lin_root(x)) + neg_sum 

        return x_out

class OledConvNE(torch.nn.Module):  
    def __init__(self, in_channels, out_channels,hop_dim,state_dim:int=5,dropout:float=0.,):  
        super().__init__()  
        edge_dim = int(in_channels/4) + hop_dim
        node_dim = in_channels
        self.node_conv = NodeConv(in_channels,out_channels,edge_dim,state_dim)
        self.edge_conv = EdgeConv(edge_dim,edge_dim,node_dim,state_dim)
        self.global_conv = globalconv(nhid=state_dim,nout=state_dim)
        
        self.dropout = dropout
        
        self.reset_parameters()  

    def reset_parameters(self):  
        pass



    def forward(self, x, edge_index, edge_attr, global_state,edge_out,origin_x,atom_index,e_idx,):   
        
        
        global_state = self.global_conv(global_state)
        x = self.node_conv( x, edge_index, edge_attr, global_state,)
        edge_out = self.edge_conv(origin_x,edge_out,atom_index,e_idx,global_state) 

        return x,global_state,edge_out




