import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

def tokenize_file(file_path: str, out_path: str, tokenizer:PreTrainedTokenizer):
    token_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            encoding = tokenizer(line,
                                 max).input_ids
            token_ids.extend(encoding)
    token_ids = np.array(token_ids)
    np.save(out_path, token_ids)
    print(f"Tokenized data saved to '{out_path}'!")


def tokenize_file_new(
    file_path: str,
    out_path: str,
    tokenizer: PreTrainedTokenizer,
    dtype=np.int64
):
    # ===============================
    # Pass 1: count total tokens
    # ===============================
    total_tokens = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_tokens += len(
                tokenizer(line, add_special_tokens=False).input_ids
            )

    print(f"Total tokens: {total_tokens:,}")

    # ===============================
    # Allocate memory-mapped array
    # ===============================
    token_ids = np.memmap(
        out_path,
        mode="w+",
        shape=(total_tokens,),
        dtype=dtype
    )

    # ===============================
    # Pass 2: write tokens
    # ===============================
    idx = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ids = tokenizer(line, add_special_tokens=False).input_ids
            n = len(ids)
            token_ids[idx:idx+n] = ids
            idx += n

    token_ids.flush()
    print(f"Tokenized data written to '{out_path}'")



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
                 shuffle: bool = False):
        self.tokenized_ids = np.load(data_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle

    def __len__(self):
        return len(self.tokenized_ids) // self.context_length // self.batch_size
        
    def get_chunk(self, start_ind):
        chunk = self.tokenized_ids[start_ind: start_ind + self.context_length + 1]
        inputs = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:])
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
            inputs_batch = torch.empty((self.batch_size, self.context_length), dtype=torch.int64)
            targets_batch = torch.empty((self.batch_size, self.context_length), dtype=torch.int64)
            for i in range(self.batch_size):
                start_ind = sample_offsets[batch_start + i]
                inputs, targets = self.get_chunk(start_ind)
                inputs_batch[i] = inputs
                targets_batch[i] = targets
            yield inputs_batch, targets_batch

class PretrainDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 context_length: int,
                 dtype: torch.dtype = torch.int64):
        '''
        data_path: str. Path to a one sequnence token ids file, should be in np.adarray format.
        '''
        super().__init__()
        self.token_ids = np.load(data_path, mmap_mode="r")
        self.context_length = context_length
        self.dtype = dtype

    def __len__(self):
        return len(self.token_ids) // (self.context_length + 1)
    
    def __getitem__(self, index):
        chunk = self.token_ids[index * (self.context_length + 1): (index + 1) * (self.context_length + 1)]
        inputs = torch.from_numpy(chunk[:-1]).to(self.dtype)
        targets = torch.from_numpy(chunk[1:]).to(self.dtype)
        return inputs, targets
        



if __name__ == "__main__":
    # x = np.random.randint(low=0, high=100, size=(50))
    # input, target = get_batch(x, batch_size=4, context_length=10)
    # print(input.dtype)
    # print(target)

    # Tokenize a .txt file
    data_path = "/root/autodl-tmp/CS336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenize_file_new(data_path,
                  out_path="../data/TinyStoriesV2-GPT4-train.npy", 
                  tokenizer=tokenizer)

    # data_path = r"data\test_data_100.npy"
    # ds = PretrainDataset(data_path, context_length=6)
    # # print(len(ds))
    # # print(ds.__getitem__(0))

    # dl = DataLoader(
    #     ds,
    #     batch_size=3,
    #     shuffle=False,
    #     num_workers=0
    # )
    # for inputs, targets in dl:
    #     print(inputs)