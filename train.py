import os
import torch
import models
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from dataset import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.LSTM_BC(learning_rate,
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

dataset = ForexDataset(files, seq_len)

# Metrics
train_accs = []
validation_accs = []
losses = []

try:
    for epoch in range(epochs):
        model.initial_hidden = model.init_hidden()

        #------------------------------------- Train loop----------------------------------- #
        dataset_done = False
        minibatch_losses = []
        minibatch_accs = []
        
        while True:
            data, labels, dataset_done = dataset.get_batch(batch_size, 'train')

            if dataset_done:
                break
            data = torch.FloatTensor(data)
            labels = torch.FloatTensor(labels)

            assert(len(data) == seq_len)

            if str(device) == 'cuda':
                data = data.cuda()
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
        train_acc_mean = np.array(minibatch_accs).mean()

        if epoch % plot_save_freq:
            losses.append(loss_mean) 
            train_accs.append(train_acc_mean)
        
        #------------------------------------- Validation loop----------------------------------- #
        # dataset_done = False
        # minibatch_accs = []

        # with torch.no_grad():
        #     while True:
        #         data, labels, dataset_done = dataset.get_batch(batch_size, 'validation')
        #         if dataset_done:
        #             break
        #         data = torch.FloatTensor(data)
        #         labels = torch.FloatTensor(labels)

        #         assert(len(data) == seq_len)

        #         if str(device) == 'cuda':
        #             data = data.cuda()
        #             labels = labels.cuda()

        #         out = model(data)

        #         # Calculating accuracy
        #         y_hat = out.cpu().detach().numpy()
        #         y = labels.cpu().numpy()

        #         acc = (y_hat > acc_threshold)
        #         acc = (acc * 1 == y) * 1
        #         acc = acc.sum() / len(y)
        #         minibatch_accs.append(acc)


        #     validation_acc_mean = np.array(minibatch_accs).mean()

        #     if epoch % plot_save_freq:
        #         validation_accs.append(validation_acc_mean)
        
        
        #------------------------------------- lr decay -------------------------------------- #
        if lr_decay_active:
            model.learning_rate_decay(epoch)

        #------------------------------------- Epoch output -------------------------------------- #
        print('\nEpoch: {}'.format(epoch))
        print('Loss: {}'.format(loss_mean))
        print('Train Accuracy: {}'.format(train_acc_mean))
        # print('Validation Acurracy: {}'.format(validation_acc_mean))
        print('Learning rate: {}'.format(model.optimizer.param_groups[0]['lr']))

except KeyboardInterrupt:
    print('se acabo')

    #------------------------------------- Loss plot -------------------------------------- #
    plt.plot(losses, label="agent reward")
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title('Loss plot')
    plt.ylabel('Loss')
    plt.xlabel('Epochs x' + str(plot_save_freq))
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

    #------------------------------------- Train acc plot -------------------------------------- #
    plt.plot(train_accs, label="agent reward")
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title('Train acc plot')
    plt.ylabel('Train accuracy')
    plt.xlabel('Epochs')
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

    #------------------------------------- Validation acc plot -------------------------------------- #
    plt.plot(validation_accs, label="agent reward")
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title('Validation acc plot')
    plt.ylabel('Validation accuracy')
    plt.xlabel('Epochs')
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()