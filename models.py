import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LSTM_BC(nn.Module):
    """
        Params:
            n_layers (L):
                number of lstm layers

            input_dim (K):
                size of each element of the sequences 

            sequence_size (n):
                size of each temporal section

            hidden_dim:
                length of hidden (memories)

            output_size:
                lenght of output layer

            drop_prob:
                probability of dropping for dropout layer
        
        Notes:
            - input shape is (n,m,K)
    """
    def __init__(self, learning_rate, n_layers, input_dim, sequence_size, hidden_dim, batch_size, output_size, device, drop_prob=0.5, lr_decay=1000):
        super(LSTM_BC, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.sequence_size = sequence_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=False)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, output_size)
        self.sigmoid = nn.Sigmoid()
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # the initial hidden units (memories) are zeros
        self.initial_hidden = self.init_hidden()

    def forward(self, x):
        hidden = self.initial_hidden

        if str(self.device) == 'cuda':
            x = x.cuda()

        lstm_out, hidden = self.lstm(x, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = lstm_out
        out = F.relu(self.fc(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.sigmoid(out)

        out = out.view(self.batch_size, -1)
        out = out[:,-1]

        return out

    def backpropagate(self,prediction,gt):
        loss = nn.BCELoss()(prediction.squeeze(), gt.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def init_hidden(self):
        hidden = (torch.FloatTensor(np.zeros((self.n_layers, self.batch_size, self.hidden_dim))).to(self.device),
                  torch.FloatTensor(np.zeros((self.n_layers, self.batch_size, self.hidden_dim))).to(self.device))
        return hidden

    def learning_rate_decay(self, epoch):
        lr = self.learning_rate * math.exp(- epoch / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    