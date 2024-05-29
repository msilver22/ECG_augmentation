from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from cnn_lightining import ECGDataModule,CNN_ecg
import matplotlib.pyplot as plt
import torch

tensorECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/tensorECG.pt'
labels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/labels.pt'
trainECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/trainECG.pt'
trainlabels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/trainlabels.pt'
testECG_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/testECG.pt'
testlabels_path = '/Users/silver22/Documents/AI trends/data/torch_dataset/testlabels.pt'

ecg_datamodule = ECGDataModule(
        train_data_path=trainECG_path,
        train_labels_path=trainlabels_path,
        test_data_path=testECG_path,
        test_labels_path=testlabels_path,
        batch_size=100
    )

cnn_model = CNN_ecg(
        lr=0.0001,
    )

cnn_model.cnn.load_state_dict(torch.load('weights.pth'))

tester = pl.Trainer(logger=False, enable_checkpointing=False)
tester.test(cnn_model, ecg_datamodule)

