"""
timesteps (M):
    number of temporal sections
"""
timesteps = 5

"""
n_layers (L):
    number of hidden units/layers
"""
n_layers = 2

"""
input_dim (K):
    size of each element of the sequences
"""
input_dim = 6

"""
subsequence_size (n):
    size of each temporal section
    TO FIX
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
batch_size = 1024
learning_rate = 0.01
lr_decay_active = False
lr_decay = 8000
epochs = 10000 

"""
dataset parameters
    dataset related parameters
"""
single_pair = True
pair = 'EURUSD'
timestep = '4h'
acc_threshold = 0.6
is_sin = False

if is_sin:
    input_dim = 1