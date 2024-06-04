import torch
import matplotlib.pyplot as plt

tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
labels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels.pt'
tensorECG_1228_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
labels_1228_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels.pt'

T = torch.load(tensorECG_path)
labels = torch.load(labels_path)

new_T = T[:,:,:7368]
new_labels = torch.zeros([labels.shape[0],labels.shape[1]*6])

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        label = labels[i,j]
        if label!=0: new_labels[i,6*j:6*j+6] = label

new_T = new_T.reshape(10, 4217, 6, 1228)
T_1228 = new_T.reshape(10, 4217*6, 1228)

print(T_1228.shape, new_labels.shape)

ecg = T_1228[0,1918*6,:]
ecg2 = T_1228[0,1919*+3,:]

print(new_labels[0,1918*6], new_labels[0,1919*6+2])

plt.figure()
plt.plot(ecg2)
plt.show()


