import os
import torch
import models
import numpy as np
from parameters import *
from dataset import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.LSTM_BC(learning_rate,
                       timesteps,
                       n_layers,
                       input_dim,
                       seq_len,
                       hidden_dim,
                       batch_size,
                       1,device,
                       drop_prob,
                       lr_decay).to(device).float()

# Dataset load
files = []
if single_pair:
    files.append('./dataset/' + timestep + '/' + pair + '_' + timestep + '.csv')
else:
    files = os.listdir('./dataset/' + timestep)

dataset = ForexDataset(files, timesteps)

# Metrics
train_accs = []
validation_accs = []
losses = []

for epoch in range(epochs):

    #------------------------------------- Train loop----------------------------------- #
    dataset_done = False
    minibatch_losses = []
    minibatch_accs = []
    

    while True:
        data, labels, dataset_done = dataset.get_batch(batch_size, 'train')

        if dataset_done:
            break
        data = [torch.FloatTensor(timestep) for timestep in data]
        labels = torch.FloatTensor(labels)

        assert(len(data) == timesteps)

        if str(device) == 'cuda':
            data = [timestep.cuda() for timestep in data]
            labels = labels.cuda()

        model.zero_grad()
        out = model(data)

        loss = model.backpropagate(out,labels)

        minibatch_losses.append(loss.item())

        # Calculating accuracy
        y_hat = out.cpu().detach().numpy()
        y = labels.cpu().numpy()

        acc = (y_hat > acc_threshold)
        acc = (acc * 1 == y) * 1
        acc = acc.sum() / len(y)
        minibatch_accs.append(acc)

    loss_mean = np.array(minibatch_losses).mean()
    losses.append(loss_mean)

    train_acc_mean = np.array(minibatch_accs).mean()
    train_accs.append(train_acc_mean)
    
    #------------------------------------- Validation loop----------------------------------- #
    dataset_done = False
    minibatch_accs = []

    with torch.no_grad():
        while True:
            data, labels, dataset_done = dataset.get_batch(batch_size, 'validation')
            if dataset_done:
                break
            data = [torch.FloatTensor(timestep) for timestep in data]
            labels = torch.FloatTensor(labels)

            assert(len(data) == timesteps)

            if str(device) == 'cuda':
                data = [timestep.cuda() for timestep in data]
                labels = labels.cuda()

            # model.zero_grad()
            out = model(data)

            # Calculating accuracy
            y_hat = out.cpu().detach().numpy()
            y = labels.cpu().numpy()

            acc = (y_hat > acc_threshold)
            acc = (acc * 1 == y) * 1
            acc = acc.sum() / len(y)
            minibatch_accs.append(acc)

        validation_acc_mean = np.array(minibatch_accs).mean()
        validation_accs.append(validation_acc_mean)

    #------------------------------------- Epoch output -------------------------------------- #
    print('\nEpoch: {}'.format(epoch))
    print('Loss: {}'.format(loss_mean))
    print('Train Accuracy: {}'.format(train_acc_mean))
    print('Validation Acurracy: {}'.format(validation_acc_mean))