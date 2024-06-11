# Augmentation techniques and generating ECG by GAN (trying to)

## [Dataset](https://github.com/msilver22/ECG_augmentation/tree/7cc301aa3ce5ea835c23c731d1771d020179a549/dataset)

We work with part of the [IRIDIA AF v1 database](https://zenodo.org/records/8405941). 

Here we represent two portions of an ecg, one of which is in Atrail-Fibrillation state and the other in normal state.

![ecg](https://github.com/msilver22/ECG_augmentation/blob/422034cd2c5f2d9d03d519c13268ef90a4ac31a3/images/ecg_example.png)

## [Detection-Model](https://github.com/msilver22/ECG_augmentation/tree/7cc301aa3ce5ea835c23c731d1771d020179a549/model)

To do detection task, we use Residual Convolutional NN to analyze 8192-samples ECGs ( 1 = atrial fibrillation).

We get around 0.95 test accuracy.
