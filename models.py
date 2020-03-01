import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, learning_rate):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.relu(self.fc(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # out.size() --> 100, 10
        return out

    def backpropagate(self, prediction, gt, kind):
        loss = nn.MSELoss()(prediction, gt)

        if kind == 'validation':
            return loss
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate_acc(self, y_pred, y):
        y_pred = y_pred.detach().numpy()
        y = y.numpy()
        answers = (y_pred > 0.5)
        matches = (answers == y)
        tp = matches.sum()
        acc = tp / len(matches)

        return acc

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, learning_rate):
        super(ANN, self).__init__()

        self.nodes = [input_dim, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, output_dim]
        self.custom_layers = []

        for i in range(1, len(self.nodes)):
            fc = nn.Linear(self.nodes[i-1], self.nodes[i])
            self.custom_layers.append(fc)
        self.custom_layers = nn.ModuleList(self.custom_layers)

        self.relu = nn.ReLU()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 100, 120, 140, 160], gamma=0.8)

    def forward(self, x):
        out = x
        for i in range(len(self.custom_layers) - 1):
            out = self.relu(self.custom_layers[i](out))
        out = self.custom_layers[-1](out)
        return out

    def backpropagate(self, prediction, gt, kind):
        loss = nn.L1Loss()(prediction, gt)

        if kind == 'validation':
            return loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate_acc(self, x, y, y_pred):
        y_pred = y_pred.detach().numpy()
        x = x.numpy()
        y = y.numpy()
        x = x.reshape(x.shape[0], -1, 4)
        last_closing = x[:, -1:, 3]
        compare_1 = (last_closing - y_pred) > 0
        compare_2 = (last_closing - y) > 0
        matches = (compare_1 == compare_2)
        tp = matches.sum()
        acc = tp / len(matches)

        return acc