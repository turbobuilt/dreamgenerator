import torch
from torch import nn
import torch.nn.functional as F
import math
from load_data_b import CustomDataLoader, DummyDataLoader
import torchaudio
import os
import glob
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from flash_attention_mha import MHA


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=.1):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.mha = MHA(d_model, num_heads, dropout=dropout, use_flash_attn=True)
        self.norm_2 = nn.LayerNorm(d_model)
        self.linear = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, src):
        src = src + self.mha(self.norm_1(src))
        src = src + self.linear(self.norm_2(src))
        return src
    
class FlashTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.05, norm_first=True, batch_first=True, activation=F.gelu):
        super(FlashTransformerEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransformerEncoderLayer(d_model,num_heads=num_heads,dim_feedforward=dim_feedforward,dropout=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, src):
        return self.layers(src)