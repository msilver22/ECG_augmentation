import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
stdECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/stdECG.pt'
afECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/afECG.pt'
ECG_augmented_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/ECG_augmented.pt'
labels_augmented_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels_augmented.pt'

std_T = torch.load(stdECG_path)
af_T = torch.load(afECG_path)

std_augmented = torch.zeros([27300,8192])
af_augmented = torch.zeros([6436,8192])

#STD AUGMENTATION
for i in tqdm(range(std_T.shape[0])):
    ecg = std_T[i,:]
    std_augmented[(7*i),:] = ecg
    blocks = ecg.reshape(64, 1, 128)
    perm1 = torch.randperm(64)
    perm2 = torch.randperm(64)
    perm3 = torch.randperm(64)
    perm4 = torch.randperm(64)
    perm5 = torch.randperm(64)
    std_augmented[(7*i)+1,:] = ecg.flip(0) #Flipped ECG
    #Permuted ECG
    permuted_blocks1 = blocks[perm1]
    aug1 = permuted_blocks1.reshape(1, 8192)
    std_augmented[(7*i)+2,:] = aug1
    permuted_blocks2 = blocks[perm2]
    aug2 = permuted_blocks2.reshape(1, 8192)
    std_augmented[(7*i)+3,:] = aug2
    permuted_blocks3 = blocks[perm3]
    aug3 = permuted_blocks3.reshape(1, 8192)
    std_augmented[(7*i)+4,:] = aug3
    permuted_blocks4 = blocks[perm4]
    aug4 = permuted_blocks4.reshape(1, 8192)
    std_augmented[(7*i)+5,:] = aug4
    permuted_blocks5 = blocks[perm5]
    aug5 = permuted_blocks5.reshape(1, 8192)
    std_augmented[(7*i)+6,:] = aug5

#AF AUGMENTATION
for i in tqdm(range(af_T.shape[0])):
    ecg = af_T[i,:]
    af_augmented[(4*i),:] = ecg
    blocks = ecg.reshape(64, 1, 128)
    perm1 = torch.randperm(64)
    perm2 = torch.randperm(64)

    af_augmented[(4*i)+1,:] = ecg.flip(0) #Flipped ECG
    #Permuted ECG
    permuted_blocks1 = blocks[perm1]
    aug1 = permuted_blocks1.reshape(1, 8192)
    af_augmented[(4*i)+2,:] = aug1
    permuted_blocks2 = blocks[perm2]
    aug2 = permuted_blocks2.reshape(1, 8192)
    af_augmented[(4*i)+3,:] = aug2


print(std_augmented.shape,af_augmented.shape)
combined_tensor = torch.cat((std_augmented, af_augmented), dim=0)
print(combined_tensor.shape)
#Create labels
labels = torch.zeros([combined_tensor.shape[0]])
labels[27300:] = 1

torch.save(combined_tensor, ECG_augmented_path)
torch.save(labels,labels_augmented_path)


"""
#Visualization
plt.figure(1, figsize=(12,7))
plt.subplot(2,1,1)
plt.plot(std_augmented[10285,:], 'k')
plt.title("STD ECG augmented")

plt.subplot(2,1,2)
plt.plot(af_augmented[3434,:], 'k')
plt.title("AF ECG augmented")
plt.tight_layout()
plt.show()
"""
    

