"""
cell_num (M):
    number of cells of the network = number of temporal sections
"""
cell_num = 10

"""
n_layers (L):
    number of layers of each cell
"""
n_layers = 1

"""
input_dim (K):
    size of each element of the sequences 
"""
input_dim = 6

"""
subsequence_size (n):
    size of each temporal section
"""
seq_len = 1

"""
hidden_dim:
    length of hidden (memories)
"""
hidden_dim = 10

"""
drop_prob:
    probability of dropping for dropout layer
"""
drop_prob = 0.5

"""
misc hyperparameters:
    you know what these are
"""
batch_size = 10
learning_rate = 0.001
lr_decay_active = False
lr_decay = 8000
epochs = 10000 