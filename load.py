import numpy as np
import os
from dataset import *

single_pair = True
pair = 'EURUSD'
timestep = '4h'
number_of_candles = 50

files = []
if single_pair:
    files.append('./dataset/' + timestep + '/' + pair + '_' + timestep + '.csv')
else:
    files = os.listdir('./dataset/' + timestep)

dataset = ForexDataset(files, number_of_candles)
data, label, _ = dataset.get_batch(64)

print(data.shape)
print(label.shape)