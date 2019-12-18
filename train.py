import torch
import torch.nn as nn
import models
import numpy as np
from parameters import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.LSTM_BC(learning_rate,
                       cell_num,
                       n_layers,
                       input_dim,
                       seq_len,
                       hidden_dim,
                       batch_size,
                       1,device,
                       drop_prob,
                       lr_decay).to(device).float()