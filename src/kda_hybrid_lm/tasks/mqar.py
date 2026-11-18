import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class MQARDateset(Dataset):
    def __init__(self, 
        num_samples:int, 
        context_length:int=32,
        num_pairs:int=4,
        num_queries:int=4,
        key_vocab_size:int=32,
        value_vocab_size:int=32,
        seed:int=42
    ):

        self.num_samples = num_samples
        self.context_length = context_length
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.key_vocab_size = key_vocab_size
        self.value_vocab_size = value_vocab_size

        self.noise_id = key_vocab_size + value_vocab_size
        self.separator_id = self.noise_id + 1
        self.vocab_size = self.separator_id + 1
        self.sequence_length = context_length + num_queries + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        return self.generate_sample(index)

def mqar_loss(logits, labels):
    return F.cross_entropy(logits, labels, ignore_index=-1)