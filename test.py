import torch
import torch.nn as nn
import models
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

input_dim = 6
hidden_dim = 10
n_layers = 1
batch_size = 10
seq_len = 1
cell_num = 10
lr = 0.5

model = models.LSTM_BC(lr,cell_num,n_layers,input_dim,seq_len,hidden_dim,batch_size,1,device).to(device).float()

X = []
for i in range(cell_num):
    inp = torch.randn(batch_size, seq_len, input_dim)
    X.append(inp)

print("\nInput:")
print("Input shapes: " , [x.shape for x in X])

# forward
out = model(X)
print("\nOutput:")
print("Output shape: ", out.shape)

# backward
gt = np.array([1,0,1,1,0,1,1,1,0,1])
gt = torch.FloatTensor(gt).cuda()

loss = model.backpropagate(out,gt)
print('loss: ', loss)