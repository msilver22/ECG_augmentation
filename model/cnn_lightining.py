import torch 
from cnn_model import CNNModel,CNNModelConfig
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import h5py
import numpy as np
from torch import Tensor
from torchvision import transforms
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score
import dataset.config as cfg

tensorECG_path = '/tensorECG.pt'
labels_path = '/labels.pt'
trainECG_path = '/trainECG.pt'
trainlabels_path = '/trainlabels.pt'
testECG_path = '/testECG.pt'
testlabels_path = '/testlabels.pt'

validation_split = 0.1
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
        sample = self.data[i,:,:]
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
        self.test_labels_path = train_labels_path

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
            self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=0
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_prediction,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
        )
    


class CNN_ecg(pl.LightningModule):
    
    def __init__(self, lr):
        super(CNN_ecg, self).__init__()

        self.cnn = CNNModel(CNNModelConfig(input_size=cfg.WINDOW_SIZE))


        self.class_loss = nn.BCELoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="binary", num_classes=2
        )

        self.save_hyperparameters()
        self.lr = lr
        self.sched_factor = 0.1
        self.sched_patience = 8

        self.f1_score = F1Score(num_classes=2, average="macro", task="binary")
        self.epoch_f1_scores = []

    def forward(self, x):
        out = self.cnn(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.sched_factor,
            patience=self.sched_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, data):
        x, labels = data
        x = x.to(device)
        labels = labels.to(device)

        probabilities = self.cnn(x)

        train_loss = self.class_loss(probabilities, labels.float())
        train_accuracy = self.accuracy(probabilities, labels.float())

        values = {"train_loss": train_loss, "train_acc": train_accuracy}
        self.log_dict(values, prog_bar=True)

        return train_loss

    def validation_step(self, data):
        x, labels = data
        x = x.to(device)
        labels = labels.to(device)

        probabilities = self.cnn(x)

        val_loss = self.class_loss(probabilities, labels.float())
        val_accuracy = self.accuracy(probabilities, labels.float())

        values = {"val_loss": val_loss, "val_acc": val_accuracy}
        self.log_dict(values, prog_bar=True)

        preds = (probabilities >= 0.5).int()
        self.f1_score(preds, labels.int())

        return val_loss

    def on_validation_epoch_end(self):
        # Compute and log the final F1 score at the end of the validation epoch
        f1 = self.f1_score.compute()
        self.log("val_f1", f1, on_epoch=True, prog_bar=True, logger=True)
        self.epoch_f1_scores.append(f1.item())

        # Reset the F1 score metric for the next epoch
        self.f1_score.reset()

    def test_step(self, data):
        x, labels = data
        x = x.to(device)
        labels = labels.to(device)

        probabilities = self.cnn(x)

        test_loss = self.class_loss(probabilities, labels.float())
        test_accuracy = self.accuracy(probabilities, labels.float())

        values = {"test_loss": test_loss, "test_acc": test_accuracy}
        self.log_dict(values, prog_bar=True)

        return test_loss

    def predict_step(self, data):
        inputs, _ = data
        inputs = inputs.to(device)
        probabilities = self.cnn(inputs)
        return torch.round(probabilities)
