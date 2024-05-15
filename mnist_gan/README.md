# GAN with PyTorch Lightning

Here we explore the power of Generative Adversarial Networks (GANs) to generate handwritten digits mimicking the MNIST dataset. This repository contains two main files that utilize PyTorch Lightning, making the model more modular and easier to maintain.

## Repository Contents

- `gan_pl_module.py`: This file contains the complete code for our GAN model implemented using PyTorch Lightning. The model architecture is designed to efficiently generate new, synthetic images of handwritten digits that resemble those found in the MNIST dataset.

- `test.py`: This script is used to test the trained generator. It generates new images and compares them with the original MNIST data to evaluate the performance and authenticity of the generated digits.
