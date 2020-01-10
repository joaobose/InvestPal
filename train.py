import os
import torch
import models
import numpy as np
from parameters import *
from dataset import *
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def plot_data():
    with torch.no_grad():
        real = []
        pred = []
        for i, data in enumerate(training_loader, 0):
            local_x, local_y = data
            y_pred = model(local_x)
            real.append(local_y.detach().numpy().squeeze())
            pred.append(y_pred.detach().numpy().squeeze())
            
        real = real[0]
        pred = pred[0]

        plt.plot(real)
        plt.plot(pred)
        plt.title('Training price prediction')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.show()

        real = []
        pred = []
        for i, data in enumerate(validation_loader, 0):
            local_x, local_y = data
            y_pred = model(local_x)
            real.append(local_y.detach().numpy().squeeze())
            pred.append(y_pred.detach().numpy().squeeze())
        real = real[0]
        pred = pred[0]

        plt.plot(real)
        plt.plot(pred)
        plt.title('Validation price prediction')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.show()


        real = []
        pred = []
        for i, data in enumerate(test_loader, 0):
            local_x, local_y = data
            y_pred = model(local_x)
            real.append(local_y.detach().numpy().squeeze())
            pred.append(y_pred.detach().numpy().squeeze())
        real = real[0]
        pred = pred[0]

        plt.plot(real)
        plt.plot(pred)
        plt.title('Test price prediction')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.show()




try:
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
    training_set = ForexDataset(files[0], number_of_candles, 'train')
    training_loader = DataLoader(dataset=training_set, **params)

    validation_set = ForexDataset(files[0], number_of_candles, 'validation')
    validation_loader = DataLoader(dataset=validation_set, **params)

    test_set = ForexDataset(files[0], number_of_candles, 'test')
    test_loader = DataLoader(dataset=test_set, **params)

    model = models.ANN(input_dim * number_of_candles, hidden_dim, layer_dim, output_dim, learning_rate)

    for epoch in range(epochs):
        train_acc = []
        total_loss = []

        for i, data in enumerate(training_loader, 0):
            local_x, local_y = data

            y_pred = model(local_x)
            loss = model.backpropagate(y_pred, local_y, 'train')
            acc = model.evaluate_acc(y_pred, local_y)

            train_acc.append(acc)
            total_loss.append(loss.item())

        train_acc = np.asarray(train_acc)
        train_acc = np.mean(train_acc)

        total_loss = np.asarray(total_loss)
        total_loss = np.mean(total_loss)
        
        with torch.no_grad():
            total_loss_valid = []
            for i, data in enumerate(validation_loader, 0):
                local_x, local_y = data
                y_pred = model(local_x)
                loss = model.backpropagate(y_pred, local_y, 'validation')
                total_loss_valid.append(loss.item())
            
            total_loss_valid = np.asarray(total_loss_valid)
            total_loss_valid = np.mean(total_loss_valid)

                
        print('Epoch {0}: Cost: {1:.8f} | Valid cost: {2:.8f}'.format(epoch, total_loss, total_loss_valid))

except KeyboardInterrupt:
    plot_data()


