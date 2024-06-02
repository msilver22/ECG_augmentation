import os
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 2)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
    


class Generator(nn.Module):
    def __init__(self, in_features=100, out_features=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32,64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, out_features),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        z = z.view(z.size(0), -1)
        return z

    
class Discriminator(nn.Module):
    def __init__(self, in_features=784, out_features=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32,out_features)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    
class GAN(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        # networks
        self.generator = Generator(in_features=self.latent_dim, out_features=784)
        self.discriminator = Discriminator(in_features=784,out_features=1)

        # loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training_step_outputs_d = []
        self.training_step_outputs_g = []
        self.d_losses = []
        self.g_losses = []

    def forward(self, z):
        return self.generator(z)
    
    def real_loss(self, predicted_outputs, loss_fn, device):

        # Targets are set to 1 here because we expect prediction to be 
        # 1 (or near 1) since samples are drawn from real dataset
        batch_size = predicted_outputs.size(0)
        targets = torch.ones(batch_size).to(device)
        real_loss = loss_fn(predicted_outputs.squeeze(), targets)
    
        return real_loss

    def fake_loss(self, predicted_outputs, loss_fn, device):

        # Targets are set to 0 here because we expect prediction to be 
        # 0 (or near 0) since samples are generated fake samples
        batch_size = predicted_outputs.size(0)
        targets = torch.zeros(batch_size).to(device)
        fake_loss = loss_fn(predicted_outputs.squeeze(), targets)
    
        return fake_loss 

    def training_step(self, batch):
        imgs, _ = batch
        imgs = imgs.to(device)

        optimizer_g, optimizer_d = self.optimizers()

        ## ----------------------------------------------------------------
        ## Train discriminator using real and then fake MNIST images,  
        ## then compute the total-loss and back-propogate the total-loss
        ## ----------------------------------------------------------------
        
        self.toggle_optimizer(optimizer_d)

        # Real MNIST images
        # Convert real_images value range of 0 to 1 to -1 to 1
        # this is required because latter discriminator would be required 
        # to consume generator's 'tanh' output which is of range -1 to 1
        imgs = (imgs * 2) - 1  
        d_real_logits_out = self.discriminator(imgs)
        d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
        
        # Fake images
        with torch.no_grad():
            # Generate a batch of random latent vectors 
            z = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
            z = torch.from_numpy(z).float().to(device)
            # Generate batch of fake images
            fake_images = self(z) 
        # feed fake-images to discriminator and compute the 
        # fake_loss (i.e. target label = 0)
        d_fake_logits_out = self.discriminator(fake_images)
        d_fake_loss = self.fake_loss(d_fake_logits_out, self.loss_fn, device)
        
        # Compute total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        self.log('d_loss', d_loss, on_step=True, prog_bar=True, logger=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        ## ----------------------------------------------------------------
        ## Train generator, compute the generator loss which is a measure
        ## of how successful the generator is in tricking the discriminator 
        ## and finally back-propogate generator loss 
        ## ----------------------------------------------------------------

        self.toggle_optimizer(optimizer_g)

        # Generate a batch of random latent vectors
        z = np.random.uniform(-1, 1, size=(self.batch_size, self.hparams.latent_dim))
        z = torch.from_numpy(z).float().to(device)       
        # Generate a batch of fake images, feed them to discriminator
        # and compute the generator loss as real_loss 
        # (i.e. target label = 1)
        fake_images = self.generator(z) 
        g_logits_out = self.discriminator(fake_images)
        g_loss = self.real_loss(g_logits_out, self.loss_fn, device)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.training_step_outputs_d.append(d_loss)
        self.training_step_outputs_g.append(g_loss)
    
    def on_train_epoch_end(self):
        d_loss_epoch = torch.stack(self.training_step_outputs_d).mean()
        g_loss_epoch = torch.stack(self.training_step_outputs_g).mean()
        self.d_losses.append(d_loss_epoch.item())
        self.g_losses.append(g_loss_epoch.item())
        self.training_step_outputs_d.clear()  # free memory
        self.training_step_outputs_g.clear()

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


def main():
    dm = MNISTDataModule()
    model = GAN()
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=50,
    )
    trainer.fit(model, dm)
    torch.save(model.generator.state_dict(), 'generator_weights.pth')
    print("[LOG] Generator weights saved.")
    d_losses = model.d_losses
    g_losses = model.g_losses
    plot_epochs = range(0, len(g_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, d_losses, "-o", label="Discriminator loss", color="blue")
    plt.plot(plot_epochs, g_losses, "-o", label="Generator loss", color="orange")
    plt.title("Loss functions over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
