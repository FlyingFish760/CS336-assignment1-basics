import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, AutoTokenizer

def tokenize_file(file_path: str, out_path: str, tokenizer:PreTrainedTokenizer):
    token_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            encoding = tokenizer(line).input_ids
            token_ids.extend(encoding)
    token_ids = np.array(token_ids)
    np.save(out_path, token_ids)
    print(f"Tokenized data saved to '{out_path}'!")


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"):
    inputs = torch.empty((batch_size, context_length), device=device, dtype=torch.int64)
    targets = torch.empty((batch_size, context_length), device=device, dtype=torch.int64)
    starting_points = torch.randint(low=0, high=len(x) - context_length, size=(batch_size,))
    for b in range(batch_size):
        starting_p = starting_points[b]
        chunk = x[starting_p: starting_p + context_length + 1]
        input_chunk = torch.from_numpy(chunk[:-1]).to(device)
        target_chunk = torch.from_numpy(chunk[1:]).to(device)
        inputs[b] = input_chunk
        targets[b] = target_chunk
    return (inputs, targets)


class CustomDataloader:
    def __init__(self, 
                 data_path: str,
                 batch_size: int, 
                 context_length: int, 
                 shuffle: bool = False,
                 device: str = "cpu"):
        self.tokenized_ids = np.load(data_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return len(self.tokenized_ids) // self.context_length // self.batch_size
        
    def get_chunk(self, start_ind):
        chunk = self.tokenized_ids[start_ind: start_ind + self.context_length + 1]
        inputs = torch.from_numpy(chunk[:-1]).to(self.device)
        targets = torch.from_numpy(chunk[1:]).to(self.device)
        return (inputs, targets)

    def load_data(self):
        '''
        Drop the last batch
        '''
        sample_offsets = np.array([ind for ind in range(0, 
                                                        len(self.tokenized_ids) - self.context_length, 
                                                        self.context_length + 1)])
        if self.shuffle:
            np.random.shuffle(sample_offsets)
        for batch_start in range(0, len(sample_offsets) - self.batch_size + 1, self.batch_size):
            inputs_batch = torch.empty((self.batch_size, self.context_length), device=self.device, dtype=torch.int64)
            targets_batch = torch.empty((self.batch_size, self.context_length), device=self.device, dtype=torch.int64)
            for i in range(self.batch_size):
                start_ind = sample_offsets[batch_start + i]
                inputs, targets = self.get_chunk(start_ind)
                inputs_batch[i] = inputs
                targets_batch[i] = targets
            yield inputs_batch, targets_batch

        



if __name__ == "__main__":
    # x = np.random.randint(low=0, high=100, size=(50))
    # input, target = get_batch(x, batch_size=4, context_length=10)
    # print(input.dtype)
    # print(target)

    data_path = r"E:\LLM\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_file(data_path,
                  out_path="../data/TinyStoriesV2-GPT4-train.npy", 
                  tokenizer=tokenizer)