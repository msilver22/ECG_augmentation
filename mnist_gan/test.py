import torch
import matplotlib.pyplot as plt
from gan_pl_module import Generator
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import math

def display_images(images, n_cols=4, figsize=(12, 6)):
    """
    Utility function to display a collection of images in a grid
    
    Parameters
    ----------
    images: Tensor
            tensor of shape (batch_size, channel, height, width)
            containing images to be displayed
    n_cols: int
            number of columns in the grid
            
    Returns
    -------
    None
    """
    plt.style.use('ggplot')
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    plt.figure(figsize=figsize)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx+1)
        image = images[idx]
        # make dims H x W x C
        image = image.permute(1, 2, 0)
        cmap = 'gray' if image.shape[2] == 1 else plt.cm.viridis
        ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])        
    plt.tight_layout()
    plt.show()

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
image_batch = next(iter(train_loader))
display_images(images=image_batch[0], n_cols=8)


latent_dim = 100  
g = Generator(in_features=100, out_features=784)
g.load_state_dict(torch.load('generator_weights.pth'))
g.eval()
z = np.random.uniform(-1, 1, size=(96, 100))
z = torch.from_numpy(z).float()
fake_images = g(z)
# Reshape and display
fake_images = fake_images.view(96, 1, 28, 28).detach()
display_images(fake_images, n_cols=8)
