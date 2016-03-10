class BatchIter:
    def __init__(self, data_array, size):
        self.data_array = data_array
        self.num_rows = data_array.shape[0]
        self.batch_size = size
        self.start_idx = 0

    def __iter__(self):
        self.start_idx = 0
        return self

    def next(self):
        if self.start_idx >= self.num_rows:
            raise StopIteration
        else:
            end_idx = min(self.start_idx + self.batch_size, self.num_rows)
            to_return = self.data_array[self.start_idx:end_idx]
            self.start_idx = end_idx
            return to_return


class PoolBatchIter:
    def __init__(self, data_array, size):
        self.data_array = data_array
        self.num_rows = len(data_array)
        self.batch_size = size
        self.start_idx = 0

    def __iter__(self):
        self.start_idx = 0
        return self

    def next(self):
        if self.start_idx >= self.num_rows:
            raise StopIteration
        else:
            end_idx = min(self.start_idx + self.batch_size, self.num_rows)
            to_return = self.data_array[self.start_idx:end_idx]
            self.start_idx = end_idx
            return to_return
