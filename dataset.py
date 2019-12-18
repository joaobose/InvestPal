from torch.utils.data import Dataset
import numpy as np
import parameters


class ForexDataset():
    def __init__(self, files, number_of_candles):
        self.samples = np.array([])
        self.labels = np.array([])
        self.dataset_length = 0

        # dimensions: (M, k, samples)
        self.training_samples = np.array([])
        self.validation_samples = np.array([])
        self.testing_samples = np.array([])

        # dimensions: (samples, 1)
        self.training_labels = np.array([])
        self.validation_labels = np.array([])
        self.test_labels = np.array([])
        
        self.train_index = 0
        self.validation_index = 0
        self.test_index = 0

        self.train_length = 0
        self.validation_length = 0
        self.test_length = 0
        
        if not parameters.is_sin:
            for file_ in files:
                tohlcv_matrix = np.loadtxt(file_, delimiter=',', dtype=float)
                i = number_of_candles

                while i < len(tohlcv_matrix):
                    # M x k samples
                    tohlcv_candles = tohlcv_matrix[i-number_of_candles:i]
                    tohlcv_next_candle = tohlcv_matrix[i]
                    label = self.get_label(tohlcv_next_candle)
                    self.samples = np.dstack((self.samples, tohlcv_candles)) if self.samples.size else tohlcv_candles
                    print(self.samples.shape)
                    self.labels = np.append(self.labels, label)
                    i += 1

        else:
            t = np.arange(0,7,0.00005)
            sin = np.sin(t)
            sin = sin.reshape(-1, 1)

            i = number_of_candles
            while i < len(sin) - 1:
                # M x k samples
                sin_samples = sin[i-number_of_candles:i]

                label = self.get_label_sin(sin[i], sin[i+1])

                self.samples = np.dstack((self.samples, sin_samples)) if self.samples.size else sin_samples
                print(self.samples.shape)
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
    
    def get_label_sin(self, current, next):
        # Get the open and close price of the last candle
        # Close price - open price
        price_difference = next - current
        if price_difference >= 0:
            return np.array([1])
        return np.array([0])

    def get_batch(self, size, kind='train'):
        dataset_end = False
        
        if kind == 'train': 
            input_batch = np.copy(self.training_samples[:,:,self.train_index:self.train_index + size])
            label_batch = np.copy(self.training_labels[self.train_index:self.train_index + size])
            self.train_index += size

            if(self.train_index >= self.train_length):
                self.train_index = 0
                dataset_end = True

        elif kind == 'validation':
            input_batch = np.copy(self.validation_samples[:,:,self.validation_index:self.validation_index + size])
            label_batch = np.copy(self.validation_labels[self.validation_index:self.validation_index + size])
            self.validation_index += size

            if(self.validation_index >= self.validation_length):
                self.validation_index = 0
                dataset_end = True
            
        else:
            input_batch = np.copy(self.testing_samples[:,:,self.test_index:self.test_index + size])
            label_batch = np.copy(self.testing_labels[self.test_index:self.test_index + size])
            self.test_index += size

            if(self.test_index >= self.test_length):
                self.test_index = 0
                dataset_end = True

        # dimensions: (n, k, samples)
        # input_batch = self.normalize(input_batch)
        
        # to: (n, m, k)
        input_batch = np.transpose(input_batch, (0, 2, 1))

        return input_batch, label_batch, dataset_end


    def normalize(self, batch):
        mean = np.mean(batch, axis=2)
        batch -= mean[:, :, np.newaxis]
        dev = np.std(batch, axis=2)
        batch /= (mean[:, :, np.newaxis] + 0.0000001)
        return batch

    def split_dataset(self):
        self.train_length = int(0.6*self.dataset_length)
        self.validation_length = int(0.2*self.dataset_length)
        self.test_length = int(0.2*self.dataset_length)

        self.training_samples = np.copy(self.samples[:,:,:int(0.6*self.dataset_length)])
        self.validation_samples = np.copy(self.samples[:,:,int(0.6*self.dataset_length):int(0.8*self.dataset_length)])
        self.testing_samples = np.copy(self.samples[:,:,int(0.8*self.dataset_length):])

        self.training_labels = np.copy(self.labels[:int(0.6*self.dataset_length)])
        self.validation_labels = np.copy(self.labels[int(0.6*self.dataset_length):int(0.8*self.dataset_length)])
        self.testing_labels = np.copy(self.labels[int(0.8*self.dataset_length):])

        if parameters.is_sin:
            self.training_samples = np.copy(self.samples)
            self.validation_samples = np.copy(self.samples)
            self.testing_samples = np.copy(self.samples)

            self.training_labels = np.copy(self.labels)
            self.validation_labels = np.copy(self.labels)
            self.testing_labels = np.copy(self.labels)

            self.train_length = self.dataset_length 
            self.validation_length = self.dataset_length 
            self.test_length = self.dataset_length

        self.samples = []
        self.labels = []