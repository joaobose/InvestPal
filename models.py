import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTM_BC(nn.Module):
    """
        Params:
            cell_num (M):
                number of cells of the network = number of temporal sections

            n_layers (L):
                number of layers of each cell

            input_dim (K):
                size of each element of the sequences 

            subsequence_size (n):
                size of each temporal section

            hidden_dim:
                length of hidden (memories)

            output_size:
                lenght of output layer

            drop_prob:
                probability of dropping for dropout layer
        
        Notes:
            - input shape is M x (m,n,K)
    """
    def __init__(self, learning_rate, cell_num, n_layers, input_dim, subsequence_size, hidden_dim, batch_size, output_size, device, drop_prob=0.5, lr_decay=1000):
        super(LSTM_BC, self).__init__()
        self.device = device
        self.cell_num = cell_num
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.subsequence_size = subsequence_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.cells = []

        for i in range(cell_num):
            cell = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
            if str(self.device) == 'cuda':
                cell = cell.cuda()
            self.cells.append(cell)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # the initial hidden units (memories) are ramdomly intialized weights
        self.initial_hidden = self.init_hidden()
        self.initial_hidden = tuple([m.data for m in self.initial_hidden])

    def forward(self, x):
        hidden = self.initial_hidden

        # LSTM cells chain
        for i in range(len(x)):
            input_t = x[i]

            if str(self.device) == 'cuda':
                input_t = input_t.cuda()

            lstm_out, hidden = self.cells[i](input_t,hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
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
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden

    def learning_rate_decay(self, epoch):
        lr = self.learning_rate * math.exp(- epoch / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    