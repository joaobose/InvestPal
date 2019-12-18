from torch.utils.data import Dataset
import numpy as np

class ForexDataset():
    def __init__(self, files, number_of_candles):
        self.samples = np.array([])
        self.labels = np.array([])
        self.index = 0

        for file_ in files:
            tohlcv_matrix = np.loadtxt(file_, delimiter=',', dtype=float)
            i = number_of_candles

            while i < len(tohlcv_matrix):
                # M x k samples
                tohlcv_candles = tohlcv_matrix[i-number_of_candles:i]
                tohlcv_next_candle = tohlcv_matrix[i]
                # print(tohlcv_candles.shape)
                label = self.get_label(tohlcv_next_candle)
                self.samples = np.dstack((self.samples, tohlcv_candles)) if self.samples.size else tohlcv_candles
                print(self.samples.shape)
                self.labels = np.append(self.labels, label)
                i += 1

    def get_label(self, next_candle):
        # Get the open and close price of the last candle
        # Close price - open price
        price_difference = next_candle[4] - next_candle[1]
        if price_difference >= 0:
            return np.array([1])
        return np.array([0])

    def get_batch(self, size):
        input_batch = self.samples[:,:,self.index:self.index + size]
        label_batch = self.labels[self.index:self.index + size]
        self.index += size
        if(self.index > len(self.samples)):
            self.index = 0
        input_batch = np.array([input_batch])
        input_batch = np.transpose(input_batch, (1, 3, 0, 2))
        return input_batch, label_batch
        
    # def test_batch(self):
    #     a = np.array([])
    #     b = np.array([[1, 1, 1, 1, 1, 1], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]])
    #     c = np.array([[2, 2, 2, 2, 2, 2], [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]])
    #     d = np.array([[3, 3, 3, 3, 3, 3], [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]])
        
    #     a = np.dstack((a,b)) if a.size else b
    #     a = np.dstack((a,c)) if a.size else b
    #     a = np.dstack((a,d)) if a.size else b
    #     a = np.transpose(a, (2, 0, 1))
    #     print(a)