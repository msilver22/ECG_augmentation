import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

labels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels.pt'
tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
stdECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/stdECG.pt'
afECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/afECG.pt'

T = torch.load(tensorECG_path)
labels = torch.load(labels_path)

std_T = torch.zeros([10000,8192])
count = 0
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        if labels[i,j] == 0 and count < 10000 :
            std_T[count,:] = T[i,j,:]
            count+=1
        if count >= 10000: break

af_T = torch.zeros([10000,8192])
count = 0
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        if labels[i,j] == 1 and count < 10000 :
            af_T[count,:] = T[i,j,:]
            count+=1
        if count >= 10000: break

torch.save(std_T,stdECG_path)
torch.save(af_T,afECG_path)
