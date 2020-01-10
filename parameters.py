number_of_candles = 50
# OHLC
input_dim = 4
# Nodes
hidden_dim = 256
# Number of stacked LSTM
layer_dim = 4
# Output after FC NN
output_dim = 1

batch_size = 64
learning_rate = 0.000005
epochs = 10000


single_pair = True
pair = 'EURUSD'
timestep = '1h'
acc_threshold = 0.5