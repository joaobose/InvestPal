from torch.utils.data import Dataset
import numpy as np

class ForexDataset():
    def __init__(self, files, number_of_candles):
        self.samples = np.array([])
        self.labels = np.array([])

        # dimensions: (M, k, samples)
        self.training_samples = np.array([])
        self.validation_samples = np.array([])
        self.testing_samples = np.array([])

        # dimensions: (samples, 1)
        self.training_labels = np.array([])
        self.validation_labels = np.array([])
        self.test_labels = np.array([])
        
        self.index = 0
        self.dataset_length = 0

        for file_ in files:
            tohlcv_matrix = np.loadtxt(file_, delimiter=',', dtype=float)
            i = number_of_candles

            while i < len(tohlcv_matrix):
                # M x k samples
                tohlcv_candles = tohlcv_matrix[i-number_of_candles:i]
                tohlcv_next_candle = tohlcv_matrix[i]
                label = self.get_label(tohlcv_next_candle)
                self.samples = np.dstack((self.samples, tohlcv_candles)) if self.samples.size else tohlcv_candles
                self.labels = np.append(self.labels, label)
                i += 1

        self.dataset_length = len(self.labels)
        self.split_dataset()
        
    def get_label(self, next_candle):
        # Get the open and close price of the last candle
        # Close price - open price
        price_difference = next_candle[4] - next_candle[1]
        if price_difference >= 0:
            return np.array([1])
        return np.array([0])

    def get_batch(self, size, kind='train'):
        dataset_end = False
        
        if kind == 'train': 
            input_batch = self.training_samples[:,:,self.index:self.index + size]
            label_batch = self.training_labels[self.index:self.index + size]
        elif kind == 'validation':
            input_batch = self.validation_samples[:,:,self.index:self.index + size]
            label_batch = self.validation_labels[self.index:self.index + size]
        else:
            input_batch = self.testing_samples[:,:,self.index:self.index + size]
            label_batch = self.testing_labels[self.index:self.index + size]
        
        self.index += size
        if(self.index >= self.dataset_length):
            self.index = 0
            dataset_end = True

        input_batch = self.normalize(input_batch)
        
        # dimensions: (1, M, k, samples)
        input_batch = np.array([input_batch])

        # to: (M, m, n, k)
        input_batch = np.transpose(input_batch, (1, 3, 0, 2))
        return input_batch, label_batch, dataset_end


    def normalize(self, batch):
        mean = np.mean(batch, axis=2)
        batch -= mean[:, :, np.newaxis]
        dev = np.std(batch, axis=2)
        batch /= (mean[:, :, np.newaxis] + 0.0000001)
        return batch

    def split_dataset(self):
        self.training_samples = self.samples[:,:,:int(0.6*self.dataset_length)]
        self.validation_samples = self.samples[:,:,int(0.6*self.dataset_length):int(0.8*self.dataset_length)]
        self.testing_samples = self.samples[:,:,int(0.8*self.dataset_length):]

        self.training_labels = self.labels[:int(0.6*self.dataset_length)]
        self.validation_labels = self.labels[int(0.6*self.dataset_length):int(0.8*self.dataset_length)]
        self.testing_labels = self.labels[int(0.8*self.dataset_length):]

        self.samples = []
        self.labels = []
        
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