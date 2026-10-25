# AI Dream Machine: Understanding VAEs from Scratch

A hands-on deep learning project exploring how neural networks learn compressed representations, reconstruct images, and generate new samples using PyTorch.

Rather than copying a tutorial, this project focuses on understanding the mathematics behind generative models and investigating their behaviour through experiments.

## Project Overview

Generative models learn patterns in data and use them to create new samples. This project begins with a simple Autoencoder and progresses to a Variational Autoencoder (VAE), exploring how probabilistic latent representations enable generation.

**Dataset:** MNIST handwritten digits  
**Framework:** PyTorch  
**Environment:** Google Colab  
**Task:** Image reconstruction and unconditional generation

## What I Built

### 1. Autoencoder

- Built an encoder–decoder neural network.
- Compressed each 28 × 28 image into a 16-dimensional vector.
- Trained the network using mean squared reconstruction error.
- Compared original images with reconstructed outputs.

### 2. Variational Autoencoder (VAE)

- Implemented a probabilistic encoder that predicts a latent mean and log variance.
- Used the reparameterization trick to sample latent vectors while preserving gradient flow.
- Combined reconstruction loss with KL-divergence regularization.
- Generated new digit-like images by sampling from a standard normal prior.

### 3. Latent-Space Exploration

- Interpolated between latent representations of different digits.
- Visualized the first two latent coordinates of test images.
- Investigated how latent representations influence decoded images.

## Experiments and Results

### Experiment 1: Latent Dimension

Three VAEs were trained for five epochs using latent dimensions of 2, 16, and 32.

| Latent Dimension | Test MSE | KL Divergence per Image |
|---:|---:|---:|
| 2 | 0.043513 | 5.5156 |
| 16 | 0.020569 | 17.5948 |
| 32 | 0.021456 | 16.8429 |

**Observation:** The 16-dimensional model achieved the lowest test MSE in this experiment. The 2-dimensional model had substantially higher reconstruction error, illustrating the limitations of a very compact representation.

### Experiment 2: KL-Divergence Weight

The beta parameter controls the strength of latent-distribution regularization.

| Beta | Test MSE | KL Divergence per Image |
|---:|---:|---:|
| 0.5 | 0.017439 | 25.8878 |
| 1.0 | 0.020429 | 17.4394 |
| 2.0 | 0.025097 | 11.8934 |

**Observation:** Lower beta produced better reconstruction in these runs, while higher beta reduced KL divergence. This illustrates a trade-off between reconstruction accuracy and latent regularization.

*These results are preliminary, based on one run per configuration with five training epochs. They do not establish a universally optimal latent dimension or beta value.*

## Mathematical Foundations

### Reconstruction Loss

For mean squared error:

\[
\mathcal{L}_{\mathrm{MSE}} =
\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
\]

### Reparameterization Trick

\[
z=\mu+\sigma\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
\]

### VAE Objective

\[
\mathcal{L}=
\mathcal{L}_{\mathrm{reconstruction}}
+D_{\mathrm{KL}}\big(q(z|x)\,\|\,p(z)\big)
\]

These components allow the VAE to learn a structured probabilistic latent space from which new samples can be generated.

## Tech Stack

- Python
- PyTorch and Torchvision
- NumPy
- Matplotlib
- Google Colab

## Key Learnings

- Neural network training, gradients and backpropagation.
- Encoder–decoder architectures and dimensionality reduction.
- Probabilistic latent variables and the reparameterization trick.
- KL-divergence regularization and the reconstruction–regularization trade-off.
- Test-set evaluation and controlled experimental comparisons.

## Limitations and Future Work

- Improve reconstruction quality and investigate training duration.
- Repeat experiments with multiple random seeds.
- Compare latent dimensions and beta values under more controlled evaluation.
- Add a visual latent-space explorer.
- Implement diffusion models from scratch as a separate, complementary project.

## Project Status

**In progress:** completing experiments, documenting findings, and improving reproducibility.

This repository is a learning project focused on understanding generative AI from its mathematical foundations to practical implementation.
