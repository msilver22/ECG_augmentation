from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from cnn_lightining import ECGDataModule,CNN_ecg
import matplotlib.pyplot as plt

tensorECG_path = '/tensorECG.pt'
labels_path = '/labels.pt'
trainECG_path = '/trainECG.pt'
trainlabels_path = '/trainlabels.pt'
testECG_path = '/testECG.pt'
testlabels_path = '/testlabels.pt'

def train(name, datamodule, model):
    """
    Trains a PyTorch Lightning model using the specified data module and logs the training process to TensorBoard.

    Parameters:
        name (str): A unique name for the training session.
        datamodule (pl.LightningDataModule): An instance of a LightningDataModule.
        model (pl.LightningModule): The model to be trained.
    """

    tb_logger = TensorBoardLogger("/lightning_logs", name=name)
    callbacks = [TQDMProgressBar(refresh_rate=10)]

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir="/.",
    )

    trainer.fit(model, datamodule)


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

train(
    name="CNN",
    datamodule=ecg_datamodule,
    model=cnn_model,
)

f1_scores_global = cnn_model.epoch_f1_scores
plot_epochs = range(0, len(f1_scores_global))
f1_scores = f1_scores_global

plt.figure(figsize=(10, 6))
plt.plot(plot_epochs, f1_scores, "-o", label="Validation F1 Score", color="blue")
plt.title("F1 Score over Epochs For CNN")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.show()

#I GET ERROR IN TEST EVALUATIONS I DON'T KNOW WHY !!!!
#tester = pl.Trainer(logger=False, enable_checkpointing=False)
#tester.test(cnn_model, ecg_datamodule)

