import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_tensor():
    t = torch.tensor([[1,0],[0,1]])
    return t


if __name__ == '__main__':
    t = create_tensor()
    t.to(device)

    print("Hello World!")
