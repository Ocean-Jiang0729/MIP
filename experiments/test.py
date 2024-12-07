# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:18:01 2024

@author: uqhjian5
"""

import torch
import torch.nn
import pickle
from test_utils import load_adj
import numpy as np
import os
import random
from argparse import ArgumentParser
import torch.optim as optim
from ..baselines.STMoE.arch_STMoE3 import STH_Experts
from test_utils import GMoE_Trainer
from test_utils import masked_mae

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)
    def inverse_transform(self, data, dim):
        return (data * self.std) + self.mean

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        
        return _wrapper()


def load_data(batch_size, data_name, forward_feature, target_feature, adj_type, device=None, normalize=True):
    data = {}
    file1 = open("datasets/"+data_name+"/data_in_{0}_out_{1}_rescale_{2}.pkl".format(12, 12, True), "rb")
    processed_data = pickle.load(file1)
    processed_data = torch.Tensor(processed_data).unsqueeze(0)
    file2 = open("datasets/"+data_name+"/index_in_{0}_out_{1}_rescale_{2}.pkl".format(12, 12, True), "rb")
    index = pickle.load(file2)
    data["adj"], adj_mx  = load_adj("datasets/"+data_name+"/adj_mx.pkl", adj_type)
    mode = ["train", "valid", "test0", "test1", "test2"]
    
    for m in mode:
        x = []
        y = []
        for i in index[m]:
            x.append(processed_data[:,i[0]:i[1],:,forward_feature])
            y.append(processed_data[:,i[1]:i[2],:,target_feature])
        data["x_"+m] = torch.cat(x,dim=0)
        data["y_"+m] = torch.cat(y,dim=0)
    mean = data['x_train'][..., 0].mean()
    std = data['x_train'][..., 0].std()
    print("mean:{},std:{}".format(mean,std))
    if normalize:
        scaler = StandardScaler(mean=mean, std=std)
        for category in mode:
            data['x_' + category] = scaler.transform(data['x_' + category])
    else:
        scaler = StandardScaler(mean=0, std=1)
    #print(data['x_train'].shape, data['y_train'].shape)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_valid'], data['y_valid'], batch_size)
    data['test0_loader'] = DataLoader(data['x_test0'], data['y_test0'], batch_size)
    data['test1_loader'] = DataLoader(data['x_test1'], data['y_test1'], batch_size)
    data['test2_loader'] = DataLoader(data['x_test2'], data['y_test2'], batch_size)
    data['scaler'] = scaler
    return data
    

def get_edge_index(adj_mx, device):
    edge_index = []
    edge_weight = []
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if adj_mx[i,j] != 0 :
                edge_index.append(torch.tensor([i,j]).reshape([2,1]))
                edge_weight.append(adj_mx[i,j])
    return torch.cat(edge_index, 1).to(device), torch.Tensor(edge_weight).to(device)


if __name__ == '__main__':
    MODEL_NAME = 'AGCRN'
    DATASET_NAME = 'PEMS03'
    BATCH_SIZE = 32
    GPUS = '2'

    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-m', '--model', default=MODEL_NAME, help='model name')
    parser.add_argument('-d', '--dataset', default=DATASET_NAME, help='dataset name')
    parser.add_argument('-g', '--gpus', default=GPUS, help='visible gpus')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    args = parser.parse_args()
    
    
    
    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cfg = 'baselines/{0}/{1}.py'.format(args.model, args.dataset)
    
    if args.model=="STGCN":
        model = cfg["MODEL"]["ARCH"]
    
    forward_feature = cfg["MODEL"]["FORWARD_FEATURES"]
    target_feature = cfg["MODEL"]["TARGET_FEATURES"]
    adj_type = "doubletransition"
    
    seq_length = cfg["MODEL"]["PARAM"]["seq_length"]
    n_layers = cfg["MODEL"]["PARAM"]["n_layers"]
    batch_size = cfg["MODEL"]["PARAM"]["batch_size"]
    in_channels = cfg["MODEL"]["PARAM"]["in_channels"]
    num_nodes = cfg["MODEL"]["PARAM"]["num_nodes"]
    hid_dim = cfg["MODEL"]["PARAM"]["hid_dim"]
    n_experts = cfg["MODEL"]["PARAM"]["n_experts"]
    k = cfg["MODEL"]["PARAM"]["k"]
    seq_predictor = cfg["MODEL"]["PARAM"]["seq_predictor"]
    dropout = cfg["MODEL"]["PARAM"]["dropout"]
    device = cfg["MODEL"]["PARAM"]["device"]
    
    lr = cfg["TRAIN"]["OPTIM"]["PARAM"]["lr"]
    weight_decay = cfg["TRAIN"]["OPTIM"]["PARAM"]["weight_decay"]

    dataloader = load_data(args.batch_size, args.dataset,forward_feature, target_feature, adj_type, device)
    scaler = dataloader['scaler']
    
    edge_index, edge_weight = get_edge_index(dataloader["adj"], device)
    model = STH_Experts(seq_length,n_layers,
                 batch_size,in_channels,num_nodes,
                 hid_dim,n_experts,k,seq_predictor,
                 edge_index,edge_weight, dropout, device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    clip = cfg["TRAIN"]["CLIP_GRAD_PARAM"]
    trainer = GMoE_Trainer(model, optimizer, masked_mae, dataloader, clip, seq_length, scaler, device)
    ckpt_dir = 'checkpoints/{0}_100'.format(args.model)
    ckpt_file = os.listdir(ckpt_dir)[0]
    ckpt_path = 'checkpoints/{0}_1/{2}/{1}/{0}_best_val_MAE.pt'.format(args.model, ckpt_file, DATASET_NAME)
    
    trainer.model.load_state_dict(torch.load('{}.pkl'.format(ckpt_path)))

    trmae, trmape, trrmse = trainer.ev_valid('train')
    vmae, vmape, vrmse = trainer.ev_valid('val')
    # tmae, tmape, trmse = trainer.ev_test('test')
    tmae0, tmape0, trmse0 = trainer.ev_valid('test0')
    print('test0', tmae0,tmape0,trmse0)
    tmae1, tmape1, trmse1 = trainer.ev_valid('test1')
    print('test1', tmae1,tmape1,trmse1)
    tmae2, tmape2, trmse2 = trainer.ev_valid('test2')
    print('test2', tmae2,tmape2,trmse2)




















