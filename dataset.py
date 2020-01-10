from torch.utils.data import Dataset
import numpy as np
import parameters
import torch

class ForexDataset(Dataset):
    def __init__(self, rootpath, number_of_candles):
        self.samples = np.array([])
        self.labels = np.array([])

        loaded_data = np.loadtxt(rootpath, delimiter=',', dtype=float)
        loaded_data = loaded_data[:int(len(loaded_data)/32), 1:5]
        i = number_of_candles

        while i < len(loaded_data):
            tohlcv_candles = loaded_data[i-number_of_candles:i]
            label = self.get_label(loaded_data[i])
            self.samples = np.dstack((self.samples, tohlcv_candles)) if self.samples.size else tohlcv_candles
            self.labels = np.append(self.labels, label)
            i += 1
            print(self.samples.shape)

        self.length = self.labels.shape[0]

        x = np.asarray(self.samples).astype(float)
        y = np.asarray(self.labels).astype(float)
        x = np.transpose(x, (2, 0, 1))
        y = y.reshape(-1, 1)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        self.samples = x
        self.labels = y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]

        return x, y
    
    def get_label(self, next_candles):
        # Get the open and close price of the last candle
        # Close prices - open price
        price_difference = next_candles[3] - next_candles[0]
        count = ((price_difference >= 0) * 1).sum()
        if count > 0:
            return np.array([1])
        return np.array([0])