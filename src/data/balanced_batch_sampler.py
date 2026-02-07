import random
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has equal positives and negatives.
    Oversamples positives if needed.
    """
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        self.labels = labels

        self.pos_indices = [i for i, y in enumerate(labels) if y == 1]
        self.neg_indices = [i for i, y in enumerate(labels) if y == 0]

        assert len(self.pos_indices) > 0, "No positive samples found!"

    def __iter__(self):
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)

        half = self.batch_size // 2
        num_batches = len(self.neg_indices) // half

        for _ in range(num_batches):
            pos_batch = random.sample(self.pos_indices, half)
            neg_batch = random.sample(self.neg_indices, half)
            yield pos_batch + neg_batch

    def __len__(self):
        return len(self.neg_indices) // (self.batch_size // 2)
