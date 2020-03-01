from oandapyV20 import API 
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.requests import MarketOrderRequest
import oandapyV20.endpoints.orders as orders
import pandas as pd
import datetime
from dateutil import parser
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import models
from parameters import *

access_token = '64ceda197abcdc5e67bc5baa2d55d5bd-27fb8b425d7fe89ca2b4c150d8c12747'
account_id = '101-011-13185020-002'
client = API(access_token)

params = {
    "count": number_of_candles + 1,
    "granularity": "H4"
}

r = instruments.InstrumentsCandles(instrument="EUR_USD",
                                   params=params)

data = client.request(r)

model = models.ANN(input_dim * number_of_candles, hidden_dim, layer_dim, output_dim, learning_rate)
model.load_state_dict(torch.load(model_path))

samples = np.array([])

for candle in data['candles']:
    candle = candle['mid']
    current_candle = np.array([candle['o'], candle['h'], candle['l'], candle['c']])
    samples = np.vstack((samples, current_candle)) if samples.size else current_candle

# Remove the last (it is still open)
samples = samples[:-1,:]
samples = samples[:,:,np.newaxis].astype(float)
x = np.transpose(samples, (2, 0, 1))
x = x.reshape((x.shape[0], -1))

local_x = torch.from_numpy(x).float()
y_pred = model(local_x)

print(y_pred)

# units positive is buy, negative is sell
# 1 is 100,000
mo = MarketOrderRequest(instrument="EUR_USD", units=0.01 * 100000)
r = orders.OrderCreate(account_id, data=mo.data)
rv = client.request(r)
print(rv)