import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper 
from core.model_utils.elements import MLP, DiscreteEncoder, Identity, VNUpdate
from torch_geometric.nn.inits import reset
from torch_geometric.nn.aggr import AttentionalAggregation
from core.model_utils.elements import globalconv,GlobalTransfer

BN = True

class GNN(nn.Module):  
    
    def __init__(self, nin, nout,nlayer, gnn_type, dropout=0,  res=True,bn=True, bias=0,):  
        super().__init__()

        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nin, nin,bias=False) for _ in range(nlayer)]) 
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nin, nout, nlayer=3, with_final_activation=False, bias=bias) if nin!=nout else Identity() 
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):

        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, x, edge_index, edge_attr, batch):

        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x += previous_x 
                previous_x = x

        x = self.output_encoder(x)
        return x

class SubgraphGNNKernel(nn.Module):
    
    
    def __init__(self, nin, nout, nlayer, gnn_types, dropout=0, 
                       hop_dim=16, 
                       bias=True, 
                       res=True,
                       pooling='add',
                       embs=(0,1,2),
                       embs_combine_mode='add',
                       mlp_layers=3,
                       subsampling=False, 
                       online=True):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']

        use_hops = hop_dim > 0
        nhid = nout // len(gnn_types)
        self.hop_embedder = nn.Embedding(20, hop_dim)
        self.gnns = nn.ModuleList()
        for gnn_type in gnn_types:
            gnn = GNN(nin+hop_dim if use_hops else nin, nhid, nlayer, gnn_type, dropout=dropout, res=res)  
            self.gnns.append(gnn)
        self.subgraph_transform = MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)
        self.context_transform =  MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)

        self.out_encoder = MLP(nout if embs_combine_mode=='add' else nout*len(embs), nout, nlayer=mlp_layers, 
                               with_final_activation=False, bias=bias, with_norm=True)

        self.use_hops = use_hops
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid()) 
        self.subsampling = subsampling

        self.dropout = dropout
        self.online = online
        self.pooling = pooling
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode

    def reset_parameters(self):
        self.hop_embedder.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data):
        
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper] 
        combined_subgraphs_edge_index = data.combined_subgraphs 
        combined_subgraphs_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        combined_subgraphs_batch = data.subgraphs_batch
        if self.use_hops:
            hop_emb = self.hop_embedder(data.hop_indicator+1)  
            combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)

        combined_subgraphs_x = torch.cat([gnn(combined_subgraphs_x, combined_subgraphs_edge_index, combined_subgraphs_edge_attr, combined_subgraphs_batch)
                                          for gnn in self.gnns], dim=-1) 

        
        if self.subsampling and self.training:
            centroid_x_selected = combined_subgraphs_x[(data.subgraphs_nodes_mapper == data.selected_supernodes[combined_subgraphs_batch])]
            subgraph_x_selected = self.subgraph_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            context_x_selected = self.context_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            if self.use_hops:
                centroid_x_selected = centroid_x_selected * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == data.selected_supernodes[combined_subgraphs_batch])]) 
                subgraph_x_selected = subgraph_x_selected * self.gate_mapper_subgraph(hop_emb)
                context_x_selected = context_x_selected * self.gate_mapper_context(hop_emb)
            subgraph_x_selected = scatter(subgraph_x_selected, combined_subgraphs_batch, dim=0, reduce=self.pooling)
            context_x_selected = scatter(context_x_selected, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)

            
            centroid_x = data.x.new_zeros((len(data.x), centroid_x_selected.size(-1)))
            centroid_x[data.selected_supernodes] = centroid_x_selected
            subgraph_x = data.x.new_zeros((len(data.x), subgraph_x_selected.size(-1)))
            subgraph_x[data.selected_supernodes] = subgraph_x_selected  
            for i in range(1, data.edges_between_two_hops.max()+1):
                
                bipartite = data.edge_index[:, data.edges_between_two_hops==i]
                scatter(centroid_x[bipartite[0]], bipartite[1], dim=0, reduce='mean', out=centroid_x)
                scatter(subgraph_x[bipartite[0]], bipartite[1], dim=0, reduce='mean', out=subgraph_x)
            
            
            context_x = context_x_selected * data.subsampling_scale.unsqueeze(-1) if self.pooling == 'add' else context_x_selected
        else:
            centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]
            subgraph_x = self.subgraph_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            context_x = self.context_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            if self.use_hops:
                centroid_x = centroid_x * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]) 
                subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
                context_x = context_x * self.gate_mapper_context(hop_emb)
            subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
            context_x = scatter(context_x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)

        x = [centroid_x, subgraph_x, context_x]
        x = [x[i] for i in self.embs]
        if self.embs_combine_mode == 'add':
            x = sum(x)
        else:
            x = torch.cat(x, dim=-1)
            
            x = self.out_encoder(F.dropout(x, self.dropout, training=self.training)) 
            
        return x


class GNNAsKernel(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer_outer, nlayer_inner, gnn_types, 
                        dropout=0, 
                        hop_dim=16, 
                        node_embedding=False, 
                        use_normal_gnn=True, 
                        bn=True, 
                        vn=False,
                        res=True, 
                        pooling='add',
                        embs=(0,1,2),
                        embs_combine_mode='add',
                        mlp_layers=3,
                        subsampling=False, 
                        online=True,
                        state_dim=None):
        super().__init__()

        nedge = int(nhid/4)
        self.state_dim = state_dim
        
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 3)  
        
        edge_emd_dim = nedge if nlayer_inner == 0 else nedge + hop_dim   
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(edge_emd_dim) if nfeat_edge is None else MLP(nfeat_edge, edge_emd_dim, 2)   
                                            for _ in range(nlayer_outer)])  
        self.edge_encoder = MLP(nfeat_edge, edge_emd_dim, 3)


        self.subgraph_layers = nn.ModuleList([SubgraphGNNKernel(nhid, nhid, nlayer_inner,gnn_types[1:], dropout, 
                                                                hop_dim=hop_dim, 
                                                                bias=not bn,
                                                                res=res, 
                                                                pooling=pooling,
                                                                embs=embs,
                                                                embs_combine_mode=embs_combine_mode,
                                                                mlp_layers=mlp_layers,
                                                                subsampling=subsampling, 
                                                                online=online) 
                                              for _ in range(nlayer_outer)])
                                              
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.norms_e = nn.ModuleList([nn.BatchNorm1d(nedge+hop_dim) if bn else Identity() for _ in range(nlayer_outer)])
        self.output_decoder = nn.Sequential(MLP(edge_emd_dim+nhid+state_dim, 128, nlayer=mlp_layers, with_final_activation=False),MLP(128, nout, nlayer=mlp_layers, with_final_activation=False))
        

        self.traditional_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid,hop_dim,state_dim) for _ in range(nlayer_outer)])  

        node_gate_nn = nn.Sequential(nn.Linear(nhid+state_dim,nhid+state_dim),nn.ELU(),
                                     nn.Linear(nhid+state_dim,nhid+state_dim),nn.ELU(),
                                     nn.Linear(nhid+state_dim,1),)
        self.node_readout = AttentionalAggregation(node_gate_nn)

        edge_gate_nn = nn.Sequential(nn.Linear(edge_emd_dim,edge_emd_dim),nn.ELU(),
                                     nn.Linear(edge_emd_dim,edge_emd_dim),nn.ELU(),
                                     nn.Linear(edge_emd_dim,1),)
        self.edge_readout = AttentionalAggregation(edge_gate_nn)

        
        self.gnn_type = gnn_types[0]
        self.dropout = dropout
        self.num_inner_layers = nlayer_inner
        self.node_embedding = node_embedding
        self.use_normal_gnn = use_normal_gnn
        self.hop_dim = hop_dim
        self.res = res
        self.pooling = pooling
        
        self.global_transfer = GlobalTransfer()

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        
        self.edge_encoder.reset_parameters()
       
       
        
        for edge_encoder, layer, norm, norm_e,old, in zip(self.edge_encoders, self.subgraph_layers, self.norms,self.norms_e, self.traditional_gnns,):
            edge_encoder.reset_parameters()
            layer.reset_parameters()
            norm.reset_parameters()
            norm_e.reset_parameters()
            old.reset_parameters()
            


    def forward(self, data):
        
        x = self.input_encoder(data.x)
        ori_edge_attr = data.edge_attr
        ori_x = x
        global_state = self.global_transfer.transfer(subgraph_size=data.subgraph_size,global_state=data.global_state.reshape(-1,2,self.state_dim),v_shape=data.v_shape)  
        
       
        edge_out = self.edge_encoder(data.edge_attr)   
        previous_x = x 


        for i, (edge_encoder, subgraph_layer, normal_gnn, norm,norm_e,) in enumerate(zip(self.edge_encoders,      
                                        self.subgraph_layers, self.traditional_gnns, self.norms, self.norms_e)):
            data.edge_attr = edge_encoder(ori_edge_attr) 
            data.x = x
            if self.use_normal_gnn:

                if self.gnn_type == 'OledConvne':
                    x,global_state,edge_out = normal_gnn(x,data.edge_index,data.edge_attr,global_state,edge_out,ori_x,data.atom_index,data.e_idx,) 
                    x = subgraph_layer(data) + x
                    edge_out = norm_e(edge_out)
            else:
                x = subgraph_layer(data)

            x = F.relu(x)
            x = norm(x)  
            x = F.dropout(x, self.dropout, training=self.training)  
            if self.res:  
                x = x + previous_x
                previous_x = x 


        x = torch.cat([x,global_state],dim=1)
        node_emb = self.node_readout(x,data.batch)
        edge_emb = self.edge_readout(edge_out,data.edge_attr_batch)

        emb = torch.cat([node_emb,edge_emb],dim=-1)

        x = self.output_decoder(emb)   

        return x
