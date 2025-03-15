# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:44:17 2024

@author: uqhjian5
"""
import torch
from torch import nn
from basicts.losses import masked_mae, masked_mae_v


#def imem_loss(prediction, out_v, target, query1, pos1, neg1, query2, pos2, neg2, null_val):
def imem_loss(prediction, out_v2,  target, lamada1, lamada2,  query1, pos1, neg1, null_val):
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
    criterion = masked_mae
    criterion_v = masked_mae_v
    #print("query{}, pos{}, neg{}".format(query1.shape, pos1.shape, neg1.shape))
    loss1 = criterion(prediction, target, null_val)
    loss = loss1
    if out_v2!=None:
        #l1 = criterion(out_v1, target, null_val).unsqueeze(0)
        l2 = criterion_v(out_v2, target, lamada1, null_val).unsqueeze(0)
        loss = loss + l2 #+ 0.01 * (loss2 + loss3) 
    loss2 = separate_loss(query1, pos1.detach(), neg1.detach())
    loss3 = compact_loss(query1, pos1.detach())
    loss += lamada2 * (loss2 + loss3)

    return loss #+ moe_loss
