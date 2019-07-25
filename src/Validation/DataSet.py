import numpy as np

class DataSet():

    def __init__(self, training_set, testing_set):
        self.datasets = [training_set, testing_set]
        self.training_set = training_set
        self.testing_set = testing_set

        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

    def get_training_fold(self,k,i):
        """ith fold is the testing fold"""
        return [ self[c] for c in range(len(self)) if (c//k) !=i]

    def get_testing_fold(self,k,i):
        """ith fold is the testing fold"""
        return [ self[c] for c in range(len(self)) if (c//k) ==i]



