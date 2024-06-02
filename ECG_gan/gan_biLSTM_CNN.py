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

tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
labels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels.pt'
trainECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/trainECG.pt'
trainlabels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/trainlabels.pt'
testECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/testECG.pt'
testlabels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/testlabels.pt'

validation_split = 0.1
BATCH_SIZE = 300
device = torch.device("mps" if torch.has_mps else "cpu")

class ECG_dataset(Dataset):

    def __init__(self, dataset_path, labels_path):
        self.data = torch.load(dataset_path)
        self.data = self.data.reshape(self.data.shape[0]*self.data.shape[1],1,self.data.shape[-1])
        self.labels = torch.load(labels_path)
        self.labels = self.labels.reshape(self.labels.shape[0]*self.labels.shape[1],1)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,i):
        sample = self.data[i,:]
        label = self.labels[i]
        return sample, label
    

class ECGDataModule(pl.LightningDataModule):

    def __init__(
            self, 
            train_data_path,
            train_labels_path,
            test_data_path,
            test_labels_path, 
            batch_size):
        super(ECGDataModule, self).__init__()

        self.batch_size = batch_size
        self.train_data_path = train_data_path
        self.train_labels_path = train_labels_path
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = ECG_dataset(
                dataset_path=self.train_data_path,
                labels_path=self.train_labels_path
            )
            train_len = train_dataset.__len__()
            validation_len = int(train_len * validation_split)

            train_val_split = [train_len - validation_len, validation_len]
            splitted_data = random_split(train_dataset, train_val_split)
            self.data_train, self.data_val = splitted_data

        if stage == "test" or stage is None:
            self.data_test = ECG_dataset(
                dataset_path=self.test_data_path,
                labels_path=self.test_labels_path
            )

        if stage == "predict" or stage is None:
            self.data_prediction = ECG_dataset(
                dataset_path=self.test_data_path,
                labels_path=self.test_labels_path
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, shuffle=True, batch_size=self.batch_size, num_workers=0
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.data_prediction,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=0,
        )

    


class Generator_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super(Generator_BiLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, z):
        z, _ = self.lstm1(z)  
        z, _ = self.lstm2(z)  
        z = self.dropout(z)
        z = self.linear(z)
        return z.reshape(z.shape[0],1,z.shape[1])

    
class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=3,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=1),
            nn.Conv1d(in_channels=3,out_channels=5,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=2),
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=3, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=5, stride=2),
            nn.MaxPool1d(kernel_size=5, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=12*253,out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
class ecg_GAN(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        hid_dim: int = 100,
        ecg_dim: int = 8192,
        lr: float = 0.00001,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE
    ):
        super(ecg_GAN,self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_dim = hid_dim
        self.output_dim = ecg_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        # networks
        self.generator = Generator_BiLSTM(
            input_dim=self.latent_dim, 
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        self.discriminator = Discriminator_CNN()

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
        ecg, _ = batch
        ecg = ecg.to(device)

        optimizer_g, optimizer_d = self.optimizers()

        ## ----------------------------------------------------------------
        ## Train discriminator using real and then fake MNIST images,  
        ## then compute the total-loss and back-propogate the total-loss
        ## ----------------------------------------------------------------
        
        self.toggle_optimizer(optimizer_d)

        d_real_logits_out = self.discriminator(ecg)
        d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
        
        # Fake images
        with torch.no_grad():
            # Generate a batch of random latent vectors 
            z = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
            z = torch.from_numpy(z).float().to(device)
            # Generate batch of fake images
            fake_ecgs = self(z) 
        d_fake_logits_out = self.discriminator(fake_ecgs)
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
        z = np.random.normal(0, 1, size=(self.batch_size, self.hparams.latent_dim))
        z = torch.from_numpy(z).float().to(device)       
        # Generate a batch of fake images, feed them to discriminator
        # and compute the generator loss as real_loss 
        # (i.e. target label = 1)
        fake_ecgs = self.generator(z) 
        g_logits_out = self.discriminator(fake_ecgs)
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
        return [opt_g, opt_d]
    
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
        train_data_path=trainECG_path,
        train_labels_path=trainlabels_path,
        test_data_path=testECG_path,
        test_labels_path=testlabels_path,
        batch_size=BATCH_SIZE
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
    plt.show()

if __name__ == '__main__':
    main()