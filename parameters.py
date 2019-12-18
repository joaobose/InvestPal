"""
n_layers (L):
    number of lstm layers
"""
n_layers = 1

"""
input_dim (K):
    size of each element of the sequences
"""
input_dim = 6

"""
sequence_size (n):
    size of each temporal section
"""
seq_len = 50

"""
hidden_dim:
    length of hidden (memories)
"""
hidden_dim = 8

"""
drop_prob:
    probability of dropping for dropout layer
"""
drop_prob = 0

"""
misc hyperparameters:
    you know what these are
"""
batch_size = 2048 #1024
learning_rate = 0.001
lr_decay_active = True # False
lr_decay = 8000 #6000
lr_step_epochs = [1300]
epochs = 10000
plot_save_freq = 50

"""
dataset parameters
    dataset related parameters
"""
single_pair = True
pair = 'EURUSD'
timestep = '4h'
acc_threshold = 0.5

"""
prediction_window:
    Max number of candles in wich the prediction can occur 
"""
prediction_window = 4

"""
is_sin:
    Is the dataset a sin(t) time series? (for model testing)
"""
is_sin = False
if is_sin:
    input_dim = 1