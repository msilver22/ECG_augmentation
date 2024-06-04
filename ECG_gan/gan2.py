import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
#from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

tensorECG_path = './tensorECG.pt'
labels_path = './labels.pt'
trainECG_path = './trainECG.pt'
trainlabels_path = './trainlabels.pt'
testECG_path = './testECG.pt'
testlabels_path = './testlabels.pt'

validation_split = 0.1
BATCH_SIZE = 32
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
    def __init__(self, train_data_path, train_labels_path, test_data_path, test_labels_path, batch_size):
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
            train_dataset = ECG_dataset(self.train_data_path, self.train_labels_path)
            train_len = train_dataset.__len__()
            validation_len = int(train_len * validation_split)
            train_val_split = [train_len - validation_len, validation_len]
            splitted_data = random_split(train_dataset, train_val_split)
            self.data_train, self.data_val = splitted_data

        if stage == "test" or stage is None:
            self.data_test = ECG_dataset(self.test_data_path, self.test_labels_path)

        if stage == "predict" or stage is None:
            self.data_prediction = ECG_dataset(self.test_data_path, self.test_labels_path)

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=True, batch_size=self.batch_size, num_workers=0)
    
    def predict_dataloader(self):
        return DataLoader(self.data_prediction, shuffle=True, batch_size=self.batch_size, num_workers=0)


class BiLSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5): # from 0.2 to 0.5
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
        # Pass through the first BiLSTM layer
        z, _ = self.lstm1(z)
        
        # Pass through the second BiLSTM layer
        z, _ = self.lstm2(z)
        
        # Apply dropout
        z = self.dropout(z)
        
        # Pass through the fully connected layer
        z = self.linear(z)
        
        # Reshape to the output format
        return z.reshape(z.shape[0],1,z.shape[1]) #z.shape[1] lunghezza sequenza (8192)

    
class Discriminator_CNN(nn.Module):
    def __init__(self, input_length=8192):
        super(Discriminator_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=120, stride=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=46, stride=3)
        
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=36, stride=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=24, stride=3)

        # Calculate the output size after convolutions and pooling to match source architecture
        conv1_output_size = (input_length - 120) // 5 + 1
        pool1_output_size = (conv1_output_size - 46) // 3 + 1
        conv2_output_size = (pool1_output_size - 36) // 3 + 1
        pool2_output_size = (conv2_output_size - 24) // 3 + 1

        # FCL
        self.fc1 = nn.Linear(in_features= 5 * pool2_output_size, out_features=25)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=25, out_features=2)  # Output 2 for binary classification
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)  # Apply softmax at the end for classification
        return x

class ecg_GAN(L.LightningModule):
    def __init__(self, latent_dim=100, hid_dim=100, ecg_dim=8192, lr=0.00001, b1=0.5, b2=0.999, batch_size=BATCH_SIZE):
        super(ecg_GAN, self).__init__()
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
        self.generator = BiLSTMGenerator(input_dim=self.latent_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
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
        batch_size = predicted_outputs.size(0)
        targets = torch.ones(batch_size).to(device)
        targets = F.one_hot(targets.to(torch.int64), num_classes=2).float()  # Convert to one-hot encoding
        real_loss = loss_fn(predicted_outputs.squeeze(), targets)
        return real_loss

    def fake_loss(self, predicted_outputs, loss_fn, device):
        batch_size = predicted_outputs.size(0)
        targets = torch.zeros(batch_size).to(device)
        targets = F.one_hot(targets.to(torch.int64), num_classes=2).float()  # Convert to one-hot encoding
        fake_loss = loss_fn(predicted_outputs.squeeze(), targets)
        return fake_loss 

    # def training_step(self, batch):
    #     ecg, _ = batch
    #     ecg = ecg.to(device)
    #     optimizer_g, optimizer_d = self.optimizers()

    #     # Train discriminator
    #     self.toggle_optimizer(optimizer_d)
    #     d_real_logits_out = self.discriminator(ecg)
    #     d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
        
    #     with torch.no_grad():
    #         z = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
    #         z = torch.from_numpy(z).float().to(device)
    #         fake_ecgs = self(z) 
    #     d_fake_logits_out = self.discriminator(fake_ecgs)
    #     d_fake_loss = self.fake_loss(d_fake_logits_out, self.loss_fn, device)
        
    #     d_loss = d_real_loss + d_fake_loss
    #     self.log('d_loss', d_loss, on_step=True, prog_bar=True, logger=True)
    #     self.manual_backward(d_loss)
    #     optimizer_d.step()
    #     optimizer_d.zero_grad()
    #     self.untoggle_optimizer(optimizer_d)

    #     # Train generator
    #     self.toggle_optimizer(optimizer_g)
    #     z = np.random.normal(0, 1, size=(self.batch_size, self.hparams.latent_dim))
    #     z = torch.from_numpy(z).float().to(device)       
    #     fake_ecgs = self.generator(z) 
    #     g_logits_out = self.discriminator(fake_ecgs)
    #     g_loss = self.real_loss(g_logits_out, self.loss_fn, device)
    #     self.log("g_loss", g_loss, prog_bar=True)
    #     self.manual_backward(g_loss)
    #     optimizer_g.step()
    #     optimizer_g.zero_grad()
    #     self.untoggle_optimizer(optimizer_g)

    #     self.training_step_outputs_d.append(d_loss)
    #     self.training_step_outputs_g.append(g_loss)
    
    def training_step(self, batch):
        ecg, _ = batch
        ecg = ecg.to(device)
        optimizer_g, optimizer_d = self.optimizers()

        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        d_real_logits_out = self.discriminator(ecg)
        d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
        
        with torch.no_grad():
            z = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
            z = torch.from_numpy(z).float().to(device)
            fake_ecgs = self(z) 
        d_fake_logits_out = self.discriminator(fake_ecgs)
        d_fake_loss = self.fake_loss(d_fake_logits_out, self.loss_fn, device)
        
        d_loss = d_real_loss + d_fake_loss
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs_d.append(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Train generator
        self.toggle_optimizer(optimizer_g)
        z = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
        z = torch.from_numpy(z).float().to(device)
        fake_ecgs = self(z) 
        d_fake_logits_out = self.discriminator(fake_ecgs)
        g_loss = self.real_loss(d_fake_logits_out, self.loss_fn, device)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs_g.append(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
    
    def on_train_epoch_end(self):
        d_loss_epoch = torch.stack(self.training_step_outputs_d).mean()
        g_loss_epoch = torch.stack(self.training_step_outputs_g).mean()
        self.d_losses.append(d_loss_epoch.item())
        self.g_losses.append(g_loss_epoch.item())
        self.training_step_outputs_d.clear()  # free memory
        self.training_step_outputs_g.clear()

    def configure_optimizers(self): 
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d]
    
def train(name, datamodule: DataLoader, model: ecg_GAN):
    tb_logger = TensorBoardLogger("./logs/lightning_logs", name=name) 
    callbacks = [TQDMProgressBar(refresh_rate=10)]

    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir="./logs/codes",
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

    # Losses visualization
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
