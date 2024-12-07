# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:44:17 2024

@author: uqhjian5
"""
import torch
from torch import nn
from basicts.losses import masked_mae


#def imem_loss(prediction, out_v, target, query1, pos1, neg1, query2, pos2, neg2, null_val):
def imem_loss(prediction, invariant_loss, target, query1, pos1, neg1, null_val):
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
    criterion = masked_mae
    #print("query{}, pos{}, neg{}".format(query1.shape, pos1.shape, neg1.shape))
    loss1 = criterion(prediction, target, null_val)
    
    loss2 = separate_loss(query1, pos1.detach(), neg1.detach())
    loss3 = compact_loss(query1, pos1.detach())
    #loss4 = separate_loss(query2, pos2.detach(), neg2.detach())
    #loss5 = compact_loss(query2, pos2.detach())
    loss = loss1 + 0.01 * (loss2 + loss3) + invariant_loss

    
    return loss