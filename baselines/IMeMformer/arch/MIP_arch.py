# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:21:53 2024

@author: uqhjian5
"""

import torch.nn as nn
import torch
from .gnn import Cheb_GNN
from .MemoryGate import MemoryGate
from .gwnet_arch import GraphWaveNet

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        #print("xx", x.shape)
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class ST_model(nn.Module):
    def __init__(self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        output_dim=1,
        model_dim=32,
        origin_adj=True,
        adpadj=True,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        use_mixed_proj=True
        ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.model_dim = model_dim
        self.feed_feed_forward_dim = feed_forward_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_mixed_proj = use_mixed_proj
        
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        #print("model_dim", self.model_dim)
        if adpadj==True and origin_adj==True:
            #print("adpadj==True and origin_adj==True")
            self.spatial_layers = nn.ModuleList(
                [
                    Cheb_GNN(self.model_dim*2, self.model_dim, cheb_k=4, dropout=self.dropout)
                    for _ in range(num_layers)
                ]
            )
        else: 
            self.spatial_layers = nn.ModuleList(
                [
                    Cheb_GNN(self.model_dim, self.model_dim, cheb_k=4, dropout=self.dropout)
                    for _ in range(num_layers)
                ]
            )

    def forward(self, x, supports, batch_size):
        for i, (spatial_layer, attn_layer_t) in enumerate(zip(self.spatial_layers, self.attn_layers_t)):
            #print("s x", x.shape)
            x = spatial_layer(x, supports)
            x = attn_layer_t(x, dim=1) #dim=1 in temporal layer 
        #print(x.shape)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out



class MIP(nn.Module):

    def __init__(
        self,
        num_nodes,
        adj,
        in_steps=12,
        out_steps=12,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=32,
        query_K = 4,
        memory_dim=32,
        memory_size=30,
        origin_feature=True,
        origin_adj=False,
        adpadj=True,
        adp_feature=True,
        lamada1 = 0.1,
        lamada2 = 0.01,
        model2_f= "concat", #"concat", "add"
        n_interv=0.15,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        alpha=0.3,
        invariant_learning=True,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.original_supports = adj
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.query_K = query_K
        self.memory_dim = memory_dim
        
        self.origin_adj = origin_adj
        self.adpadj = adpadj
        self.adp_feature = adp_feature
        self.lamada1 = lamada1
        self.lamada2 = lamada2
        self.model2_f = model2_f
        self.n_interv = n_interv
        self.origin_feature = origin_feature
        if origin_feature==True and adp_feature==True:
            self.model_dim = (
                input_embedding_dim+
                memory_dim
            )
            
        else:
            self.model_dim = memory_dim        
                
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.alpha = alpha
        self.invariant_learning = invariant_learning

        self.input_proj = nn.Linear(input_dim, self.input_embedding_dim)
        
        self.gate = MemoryGate(num_nodes=num_nodes,seq_length=in_steps,
                               mem_hid = memory_dim, 
                               input_dim = input_embedding_dim, 
                               query_K=query_K, memory_size=memory_size) #query_K=query_K, 
        
        self.model = ST_model(num_nodes,in_steps,out_steps, output_dim,
                              self.model_dim, origin_adj,adpadj,feed_forward_dim,num_heads,
                              num_layers, dropout,use_mixed_proj)
        #print("AAA", self.model_dim)
        if self.model2_f=="concat":
            model2_dim = self.model_dim+memory_dim
        if self.model2_f=="add":
            model2_dim = self.model_dim
        self.model2 = GraphWaveNet(num_nodes=self.num_nodes, dropout=dropout, 
                                   supports=self.original_supports, gcn_bool=True,
                                   addaptadj=True, aptinit=None,
                                   in_dim=model2_dim, out_dim=out_steps, 
                                   residual_channels=32,dilation_channels=32, 
                                   skip_channels=256, end_channels=512,
                                   kernel_size=2, blocks=4, layers=self.num_layers)
# =============================================================================
#         
#         self.model2 = ST_model(num_nodes,in_steps,out_steps, output_dim,
#                               self.model_dim, adpadj,feed_forward_dim,num_heads,
#                               num_layers, dropout,use_mixed_proj)
# 
# =============================================================================
    #def forward(self, history_data):
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        #print("###################################")
        #print(history_data[0,:,120,:])
        x = history_data
        #print(x[0,:, 0,0], future_data[0,:, 0,0])
        batch_size = x.shape[0]

        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        
        x_h = x#[:,-1,:,:]
        #value, query1, pos1, neg1, query2, pos2, neg2 = self.gate.query_mem(x_h)
        

        if self.query_K==1:
            value1, value2,  query1, pos1, neg1 = self.gate.query_mem(x_h)
        else:
            value1, value2, query1, pos1, neg1 = self.gate.query_mem3(x_h)

        #print(value.shape)
        if self.adpadj==True:
            supports = self.gate.get_adj() 
            if self.origin_adj==True:
                self.supports = self.original_supports+supports
            else:
                self.supports = supports
        else:
            self.supports = self.original_supports

        if self.origin_feature==True and  self.adp_feature==True:
            features = [x_h, value1]
            x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        if self.origin_feature==True and self.adp_feature==False:
            x = x_h
        if self.origin_feature==False and self.adp_feature==True:
            x = value1
        #print("input x3", x.shape)
        
        #print("supports", len(self.supports))
        out = self.model(x,self.supports, batch_size)
        #print(out.shape)
        
        if self.training and self.invariant_learning==True:
            #print("training")
            if self.query_K==1:
                f = self.gate.intervention(value2, intervention_rates=self.n_interv)
            else:
                f = self.gate.query_variant3(x_h, intervention_rates=self.n_interv)

            #out_v1=self.model2(torch.cat([x,value2], dim=-1))
            out_v2=self.model2(torch.cat([x,f], dim=-1))
            
            return {'prediction': out,'out_v2':out_v2,   #,'out_v1':out_v1
                    'lamada1':self.lamada1, 'lamada2':self.lamada2,
                     'query1': query1,'pos1': pos1, 'neg1': neg1}
        else:
            return {'prediction': out,'out_v1':None,'out_v2':None, 
                    'lamada1':0, 'lamada2':0,
                    'query1': query1,'pos1': pos1, 'neg1': neg1}
    
if __name__ == "__main__":
    seed = 2
    torch.manual_seed(seed)
    n = 150
    k = 2
    l = 12
    n_experts = 3
    device = "cuda:0"
    
    x = torch.zeros(8,l,n,3).cuda()
    s = [torch.randn(n,n).cuda(), torch.randn(n,n).cuda()]
    model = IMeMformer(n, adj=s, memory_dim=32,adpadj=True).cuda()
    #print(model)
    y = model(x)
    
# =============================================================================
#     print("Model's Parameters:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")
# 
# =============================================================================