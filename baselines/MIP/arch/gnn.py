# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:09:30 2024

@author: uqhjian5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cheb_GNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, dropout):
        super(Cheb_GNN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        
        self.dropout1 = nn.Dropout(dropout)
        
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            support = support.to(x.device)
            #print("support", support.shape)
            x_g.append(torch.matmul(support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        #print("x:{}, weights:{}".format(x_g.shape, self.weights.shape))
        x_gconv = torch.matmul(x_g, self.weights) + self.bias  # b, N, dim_out
        return self.dropout1(x_gconv)
    
class Cheb_GNN2(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, dropout):
        super(Cheb_GNN2, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        
        self.dropout1 = nn.Dropout(dropout)
        
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(support.device), support]
            for k in range(2, self.cheb_k):
                #print('1', support.shape, support_ks[-1].shape)
                a = torch.matmul(2 * support, support_ks[-1])
                #print('2', a.shape, support_ks[-2].shape)
                support_ks.append(a - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            support = support.to(x.device)
            #print("support", support.shape, x.shape)
            x_g.append(torch.einsum('bnn,blnd->blnd', support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        #print("x:{}, weights:{}".format(x_g.shape, self.weights.shape))
        x_gconv = torch.matmul(x_g, self.weights) + self.bias  # b, N, dim_out
        return self.dropout1(x_gconv)
    
if __name__ == "__main__":
    seed = 2
    torch.manual_seed(seed)
    n = 7

    device = "cuda:0"
    
    x = torch.zeros(8,10,n,2).cuda()
    A = torch.randn(8,n,n).cuda()
    
    
    
    gnn2 = Cheb_GNN(2, 6, 3, 0.3).cuda()
    y2 = gnn2(x, [A])
    print("y2", y2.shape)
    
    
