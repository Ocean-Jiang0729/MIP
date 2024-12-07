# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:37:23 2024

@author: uqhjian5
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import random
#from .gnn import Cheb_GNN



class MemoryGate(nn.Module):
    """
    Input
     - input: B, N, T, in_dim, original input
     - hidden: hidden states from each expert, shape: E-length list of (B, N, T, C) tensors, where E is the number of experts
    Output
     - similarity score (i.e., routing probability before softmax function)
    Arguments
     - mem_hid, memory_size: hidden size and total number of memroy units
     - sim: similarity function to evaluate routing probability
     - nodewise: flag to determine routing level. Traffic forecasting could have a more fine-grained routing, because it has additional dimension for the roads
        - True: enables node-wise routing probability calculation, which is coarse-grained one
    """
    def __init__(self, num_nodes,seq_length, mem_hid = 32, input_dim = 3, query_K=3, memory_size = 30, tau=0.8, sim = nn.CosineSimilarity(dim = -1), nodewise = False, ind_proj = True, attention_type = 'attention'):
        super(MemoryGate, self).__init__()
        self.num_nodes = num_nodes
        self.mem_hid = mem_hid
        self.memory_size = memory_size
        self.tau = tau
        self.attention_type = attention_type
        self.sim = sim
        self.nodewise = nodewise

        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))
        self.ln1 = nn.LayerNorm(mem_hid)
        self.ln2 = nn.LayerNorm(mem_hid)

        self.linear1 = nn.Linear(input_dim, mem_hid)
        self.We1 = nn.Parameter(torch.empty(num_nodes, memory_size))
        self.We2 = nn.Parameter(torch.empty(num_nodes, memory_size))
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.zeros_(p)
    
    def query_mem(self, input):
        #B, N, D = input.size()
        mem = self.memory
        #print("mem", mem.shape)
        #query1 = torch.matmul(input, self.input_query1)
        query1 = self.linear1(input)
        #query1 = self.input_query1(input, supports)
        #print("query1:", query1.shape)
        self.energy = torch.matmul(query1, mem.T)
        self.score1 = torch.softmax(self.energy, dim = -1)
        #print("score:", score1.shape)
        value1 = torch.matmul(self.score1, mem)
        score2 = torch.softmax(-1*self.energy, dim = -1)
        value2 = torch.matmul(score2, mem)
        #print("value:", value1.shape)
        _, ind1 = torch.topk(self.score1, k=2, dim=-1)
        #print("index1", ind1.shape)
        #print("memory:", self.memory.shape)
        pos1 = self.memory[ind1[:, :, :, 0]] # B, N, d
        neg1 = self.memory[ind1[:, :, :, 1]] # B, N, d
        #print("query:{}, pos:{}, neg:{}".format(query.shape, pos.shape, neg.shape))

        #return self.ln1(value1), query1, pos1, neg1
        return value1, value2, query1, pos1, neg1
    
    def intervention(self, value2, intervention_rates=0.25):
        
        # 设定参数
        num_nodes = value2.size(2)
        num_time_steps = value2.size(1)
        num_selected_nodes = int(num_nodes * intervention_rates/2)
        
        # 随机选取节点
        selected_nodes1 = torch.tensor(random.sample(range(num_nodes), num_selected_nodes))
        selected_nodes2 = torch.tensor(random.sample(range(num_nodes), num_selected_nodes))
        #print("Selected nodes1:", selected_nodes1)
        #print("Selected nodes2:", selected_nodes2)
        
        # 为每个选定的节点随机选择两个不同的时间片
        time_pairs = torch.tensor([random.sample(range(num_time_steps), 2) for _ in range(num_selected_nodes)])
        #print("Time pairs:", time_pairs.shape)
        
        # 使用高级索引进行交换
        time1_indices = time_pairs[:, 0]
        time2_indices = time_pairs[:, 1]
        
        # 选取需要交换的特征
        features_time1 = value2[:, time1_indices, selected_nodes1, :].clone()
        features_time2 = value2[:, time2_indices, selected_nodes2, :].clone()
        
        # 交换特征
        value2[:, time1_indices, selected_nodes1, :] = features_time2
        value2[:, time2_indices, selected_nodes2, :] = features_time1
        #print("value2", value1.shape)
        #return self.ln2(value1)
        return value2
    
    def query_variant4(self, input, intervention_rates=0.15):  #add noise
        #B, N, D = input.size()
        mem = self.memory
        
        #print("score2:", score2.shape)
        
        size1 = [1, 1, self.num_nodes, 1]
        noise1 = torch.normal(mean=0, std=0.1, size=size1).to(mem.device)
        sparsity = intervention_rates  # 50% 的元素将被设置为0

        # 生成掩码矩阵
        mask1 = torch.rand(size1) > sparsity  # 小于sparsity的为True，即这些位置将被置为0
        #print(mask1)
        # 应用掩码
        noise1[mask1] = 0
        #print(noise1)
        
        score1 = torch.softmax(-1*(self.energy1+noise1), dim = -1)
        #print("score1:", score1.shape)
        #score2 = torch.softmax(-1*(self.energy2+noise2), dim = -1)
        
        value1 = torch.matmul(score1, mem)
        
        return self.ln2(value1)

    def query_variant3(self, input, intervention_rates=0.15):
        #B, N, D = input.size()
        mem = self.memory
        score1 = torch.softmax(-1*self.energy, dim = -1)
        value1 = self.ln2(torch.matmul(score1, mem))
        #print("value12", value1.shape)
        # 设定参数
        num_nodes = value1.size(4)
        num_time_steps = value1.size(1)
        num_selected_nodes = int(num_nodes * intervention_rates/2)
        
        # 随机选取节点
        selected_nodes1 = torch.tensor(random.sample(range(num_nodes), num_selected_nodes))
        selected_nodes2 = torch.tensor(random.sample(range(num_nodes), num_selected_nodes))
        #print("Selected nodes1:", selected_nodes1)
        #print("Selected nodes2:", selected_nodes2)
        
        # 为每个选定的节点随机选择两个不同的时间片
        #print(num_time_steps, num_selected_nodes)
        time_pairs = torch.tensor([random.sample(range(num_time_steps), 2) for _ in range(num_selected_nodes)])
        #print("Time pairs:", time_pairs)
        
        # 使用高级索引进行交换
        time1_indices = time_pairs[:, 0]
        time2_indices = time_pairs[:, 1]
        
        # 选取需要交换的特征
        features_time1 = value1[:, time1_indices, :, selected_nodes1, :].clone()
        features_time2 = value1[:, time2_indices, :, selected_nodes2, :].clone()
        
        # 交换特征
        value1[:, time1_indices, :, selected_nodes1, :] = features_time2
        value1[:, time2_indices, :, selected_nodes2, :] = features_time1
        #print("value1", value1.shape)
        value1 = self.out_proj2(value1.permute(0,1,3,2,4).reshape(self.B,self.L,self.N,-1))
        #print("value2", value1.shape)
        return value1
    
    def get_adj(self):
        
        node_embeddings1 = torch.matmul(self.We1, self.memory)
        node_embeddings2 = torch.matmul(self.We2, self.memory)
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T))/0.8, dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T))/0.8, dim=-1)
        supports = [g1, g2]
        
        return supports
    
    def get_adj2(self, x):
        b, l, n, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0,2,1,3).reshape(b,n,l*d)
        xq = torch.matmul(x, self.Wq)
        xk = torch.matmul(x, self.Wk).permute(0,2,1)
        qk = torch.matmul(xq, xk)/(d**(1/2))
        A = torch.softmax(qk, dim=-1)
        return [A]



    def reset_queries(self):
        with torch.no_grad():
            for p in self.hid_query:
                nn.init.xavier_uniform_(p)
            nn.init.xavier_uniform_(self.input_query)
    
    def reset_params(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in "We1 We2 memory".split():
                    continue
                else:
                    nn.init.xavier_uniform_(p)


if __name__ == "__main__":
    seed = 2
    torch.manual_seed(seed)
    n = 50
    k = 2
    l = 12
    n_experts = 3
    device = "cuda:0"
    
    x = torch.zeros(8,l,n,3).cuda()
    gate = MemoryGate(num_nodes=n, seq_length=l).cuda()
    value, query, pos, neg = gate.query_mem3(x)
    print("value{}, query{}, pos{}, neg{}".format(value.shape, query.shape, pos.shape, neg.shape))
    v = gate.query_variant(x, 0.5)
    print("m", value.shape)
    print("v", v.shape)
    a = gate.get_adj2(value)
    print('a', a[0][0,2])



