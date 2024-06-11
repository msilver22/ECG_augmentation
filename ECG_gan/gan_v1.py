import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

stdECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/stdECG.pt'
afECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/afECG.pt'

T = torch.load(stdECG_path)

BATCH_SIZE = 16
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

class ECG_gan_dataset(Dataset):

    def __init__(self, af = False):
        if af == True:
            self.data = torch.load(afECG_path)
        else: 
            self.data = torch.load(stdECG_path)

        self.data = self.data.reshape(self.data.shape[0],1,self.data.shape[-1])
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,i):
        sample = self.data[i,:]
        return sample
    

class ECGDataModule(pl.LightningDataModule):

    def __init__(
            self,  
            batch_size,
            af = False
        ):
        super(ECGDataModule, self).__init__()

        self.batch_size = batch_size
        self.af = af

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = ECG_gan_dataset(self.af)
            self.data_train = train_dataset


    def train_dataloader(self):
        return DataLoader(
            self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=0
        )

    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=8192//64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        
        self.upsample = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(in_channels=4,out_channels=1,kernel_size=3,stride=1,padding='same'),
            nn.Tanh()
        ) 
      
        
    def forward(self, z):
        z,_ = self.lstm(z)
        z = z.reshape(z.shape[0],z.shape[-1],z.shape[1])       
        z = self.upsample(z)
        return z
    
class BiLSTMGenerator(nn.Module):
    def __init__(self, input_dim=8192, hidden_dim=100, output_dim=8192, dropout_prob=0.5): # from 0.2 to 0.5
        super(BiLSTMGenerator, self).__init__()
        # BiLSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  # Bidirectional output
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z):
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.dropout(z)
        z = self.linear(z)
        
        return z.reshape(z.shape[0],1,z.shape[1]) #z.shape[1] lunghezza sequenza (8192)

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding='same'),
            nn.Flatten(),
            nn.Linear(in_features=1024*128,out_features=1),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        x = self.model(x)
        return x
    
class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=10,kernel_size=120,stride=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=46,stride=3),
            nn.Conv1d(in_channels=10,out_channels=5,kernel_size=36,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=24,stride=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=5*47,out_features=25),
            nn.Softmax(),
            nn.Linear(25,1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

    
class ecg_GAN(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.00002,
        factor: float = 0.1,
        patience: int = 5,
        batch_size: int = BATCH_SIZE
    ):
        super(ecg_GAN,self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.lr = lr
        self.sched_factor = factor
        self.sched_patience = patience

        # networks
        self.generator = Generator()
        self.discriminator = Discriminator()

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
    
    def calculate_gradient_penalty(self, batch_size, real_ecgs, fake_ecgs):

        alpha = torch.randn((batch_size, 1, 1), device=device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real_ecgs + ((1 - alpha) * fake_ecgs)).requires_grad_(True)

        model_interpolates = self.discriminator(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def training_step(self, batch):
        ecg = batch
        batch_size = ecg.shape[0]
        ecg = ecg.to(device)

        optimizer_g, optimizer_d = self.optimizers()

        ## ----------------------------------------------------------------
        ## Train discriminator using real and then fake MNIST images,  
        ## then compute the total-loss and back-propogate the total-loss
        ## ----------------------------------------------------------------
        
        self.toggle_optimizer(optimizer_d)

        d_real_logits_out = self.discriminator(ecg)
        d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
        #errD_real = torch.mean(d_real_logits_out)

        
        # Fake images
        with torch.no_grad():
            # Generate a batch of random latent vectors 
            z = torch.randn(size=(self.batch_size,256,16),device = device)
            # Generate batch of fake images
            fake_ecgs = self(z) 
        d_fake_logits_out = self.discriminator(fake_ecgs)
        d_fake_loss = self.fake_loss(d_fake_logits_out, self.loss_fn, device)
        #errD_fake = torch.mean(d_fake_logits_out)

        #gradient_penalty = self.calculate_gradient_penalty(
         #   batch_size,
          #  ecg, 
          #  fake_ecgs
        #)
        
        # Compute total discriminator loss
        d_loss = d_fake_loss + d_real_loss
        #errD = -errD_real + errD_fake + gradient_penalty * 10
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

        for i in range(2):

            # Generate a batch of random latent vectors
            z = torch.randn(size=(self.batch_size,256,16),device = device)  
            # Generate a batch of fake images, feed them to discriminator
            # and compute the generator loss as real_loss 
            # (i.e. target label = 1)
            fake_ecgs = self.generator(z) 
            g_logits_out = self.discriminator(fake_ecgs)
            g_loss = self.real_loss(g_logits_out, self.loss_fn, device)
            #errG = -torch.mean(g_logits_out)
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
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr*10, betas=(0.5,0.99))
        scheduler_g = ReduceLROnPlateau(
            optimizer_g,
            mode="min",
            factor=self.sched_factor,
            patience=self.sched_patience,
        )
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5,0.99))
        scheduler_d = ReduceLROnPlateau(
            optimizer_d,
            mode="max",
            factor=self.sched_factor,
            patience=self.sched_patience,
        )
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
    
def train(name, datamodule: DataLoader, model: ecg_GAN):
    tb_logger = TensorBoardLogger("/Users/silver22/Documents/AI trends/lightning_logs", name=name)
    callbacks = [TQDMProgressBar(refresh_rate=10)]

    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir="/Users/silver22/Documents/AI trends/codes",
    )

    trainer.fit(model=model, datamodule=datamodule)
    torch.save(model.generator.state_dict(), 'weights.pth')
    print("[LOG] Generator weights saved.")


def main():
    ecg_datamodule = ECGDataModule(
        batch_size=BATCH_SIZE,
        af=False
    )
    ecg_model = ecg_GAN()
    train(
        name="GAN",
        datamodule=ecg_datamodule,
        model=ecg_model,
    )

    #Losses visualization
    d_losses = ecg_model.d_losses
    g_losses = ecg_model.g_losses
    plot_epochs = range(0, len(g_losses))
    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, d_losses, "-o", label="Discriminator loss", color="blue")
    plt.plot(plot_epochs, g_losses, "-o", label="Generator loss", color="orange")
    plt.title("Loss functions over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig('GAN_losses.png')
    plt.show()

if __name__ == '__main__':
    main()
