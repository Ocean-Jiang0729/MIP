# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:46:02 2024

@author: uqhjian5
"""

import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import load_adj

from .arch import IMeMformer
from .loss import imem_loss

CFG = EasyDict()
def select_nodes(adj_list,n):
    adj = []
    for a in adj_list:
        adj.append(a[0:n,0:n])
    return adj

# ================= general ================= #
CFG.DESCRIPTION = "IMeMformer model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "NYCBike"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 6
CFG.DATASET_OUTPUT_LEN = 6
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 4
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
N = 128
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "IMeMformer"
CFG.MODEL.ARCH = IMeMformer
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")
adj_mx = select_nodes(adj_mx,N)
CFG.MODEL.PARAM = {
    "num_nodes" : N,
    "in_steps" : CFG.DATASET_INPUT_LEN,
    "out_steps" : CFG.DATASET_OUTPUT_LEN,
    "adj" : [torch.tensor(i) for i in adj_mx],
    "input_dim" : 2,
    "origin_feature": False,
    "origin_adj": True,
    "adpadj" : True,
    "adp_feature" : True,
    "invariant_learning" : True,
    "lamada1" : 0.5,  #invariant learning
    "lamada2" : 0.05,
    "model2_f" : "concat", #"concat", "add"
    "num_layers" : 3, 
    "memory_size" : 30,
    "memory_dim" : 32,
    "query_K" : 1,
    "n_interv" : 0.25, 
    "dropout" : 0.1,
    "alpha" : 0.1
}
CFG.MODEL.FORWARD_FEATURES = [0,1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = imem_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.003,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20,30, 40,50, 60,70, 80, 90, 100, 110,120, 130, 140],
    "gamma": 0.8
}

# ================= train ================= #
# CFG.TRAIN.CLIP_GRAD_PARAM = {
#     "max_norm": 5.0
# }
CFG.TRAIN.NUM_EPOCHS = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)]),
    CFG.DATASET_NAME
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [CFG.DATASET_OUTPUT_LEN]