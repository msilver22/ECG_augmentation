import torch
import matplotlib.pyplot as plt
from gan_biLSTM_CNN import Generator_BiLSTM
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import math

tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'

T = torch.load(tensorECG_path)

af_ecg = T[10,1935,:]
std_ecg = T[10,1000,:]

latent_dim = 100  
g = Generator_BiLSTM(
            input_dim=latent_dim, 
            hidden_dim=100,
            output_dim=8192
        )
g.load_state_dict(torch.load('weights.pth'))
g.eval()
z_start = np.random.normal(0, 1, size= (10,latent_dim))
z = torch.from_numpy(z_start).float()
fake_ecgs = g(z)

ecg = fake_ecgs[0,0,:]
ecg = ecg.detach().numpy()

plt.figure(figsize=(12,7))

plt.subplot(3,1,1)
plt.plot(af_ecg,'k')
plt.title("AF ECG")

plt.subplot(3,1,2)
plt.plot(std_ecg,'k')
plt.title("STD ECG")

plt.subplot(3,1,3)
plt.plot(ecg,'k')
plt.title("FAKE ECG")

plt.tight_layout()
plt.show()



