import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from core.model_utils.elements import MLP,OledConvNE
import torch.nn.functional as F



class CGConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = gnn.CGConv(channels=(nin, nout), dim=nin, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class NNConv(nn.Module):  
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nin*nout, 2, False)
        self.layer = gnn.NNConv(in_channels=nin, out_channels=nout, nn=self.nn, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class TransformerConv(nn.Module): 
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.TransformerConv(in_channels=nin, out_channels=nout//nhead, heads=nhead, edge_dim=int((nin-16)/4+16), bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GINEConv(nn.Module):  
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GATConv(nn.Module):  
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)

class ResGatedGraphConv(nn.Module):  
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = gnn.ResGatedGraphConv(nin, nout, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        
        
        self.layer = gnn.GCNConv(nin, nout, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)
        


    
class OledConvne(nn.Module):

    def __init__(self,nin,nout,hop_dim,state_dim:int=5,bias=False):
        super().__init__()
        self.layer = OledConvNE(nin,nout,hop_dim,state_dim)

    def reset_parameters(self):
        self.layer.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr, global_state,edge_out,origin_x,atom_index,e_idx,): 
        
        return self.layer(x, edge_index, edge_attr, global_state,edge_out,origin_x,atom_index,e_idx,)  
    