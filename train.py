import os
import torch
import models
import numpy as np
from parameters import *
from dataset import *
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Dataset load

files = []
if single_pair:
    files.append('./dataset/' + timestep + '/' + pair + '_' + timestep + '.csv')
else:
    files = os.listdir('./dataset/' + timestep)

params = {'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2}

# Create Dataset
training_set = ForexDataset(files[0], number_of_candles)
training_loader = DataLoader(dataset=training_set, **params)

model = models.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, learning_rate)

for epoch in range(epochs):
    train_acc = []
    total_loss = []

    for i, data in enumerate(training_loader, 0):
        local_x, local_y = data

        y_pred = model(local_x)
        loss = model.backpropagate(y_pred, local_y)
        acc = model.evaluate_acc(y_pred, local_y)

        train_acc.append(acc)
        total_loss.append(loss.item())

    train_acc = np.asarray(train_acc)
    train_acc = np.mean(train_acc)

    total_loss = np.asarray(total_loss)
    total_loss = np.mean(total_loss)
    print('Epoch {0}: Cost: {1:.4f} | Training acc: {2:.4f}'.format(epoch, total_loss, train_acc,))