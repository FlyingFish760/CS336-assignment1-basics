import numpy as np
import torch
from torch import Tensor

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


# class CustomDataloader:
#     '''
    
#     '''
#     def __init__(self, 
#                  data: np.ndarray, 
#                  context_length: int):
#         self.data = data
#         self.

#     def batch_data(self,):
        

#     def __get_item__(self,):

#     def __len__(self,):
        
#     def get_batch(self,):

#     def load_data(self, )


if __name__ == "__main__":
    x = np.random.randint(low=0, high=100, size=(50))
    input, target = get_batch(x, batch_size=4, context_length=10)
    print(input.dtype)
    print(target)