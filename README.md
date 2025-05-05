# Bayesian Conditional Tabular GAN (Bayesian CTGAN)

## Overview

This repository implements a Bayesian Conditional Tabular GAN (Bayesian CTGAN), a novel extension of the Conditional Tabular GAN (CTGAN) for generating synthetic tabular data. The key innovation is the integration of a Bayesian Generator with a residual architecture, which models weights as random variables to enhance output diversity and enable uncertainty quantification. The model uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) framework, augmented with conditional generation for mixed continuous and discrete columns, and a PacGAN approach to prevent mode collapse.

The implementation is designed for tabular datasets with mixed data types, addressing privacy-preserving data sharing in domains like finance and healthcare. This README explains the theoretical foundations, including the learning of weight distributions, the Discriminator and Generator losses, and provides instructions for running the code.

## Theoretical Foundations

### Motivation

Synthetic tabular data generation is critical for privacy-preserving data sharing and augmentation. CTGAN [Xu et al., 2019] is a state-of-the-art model that uses WGAN-GP and conditional generation to handle mixed data types. However, its deterministic generator limits output diversity and lacks uncertainty quantification. The Bayesian CTGAN addresses these limitations by:

- Modeling generator weights as random variables using variational inference, increasing diversity and enabling uncertainty estimates.
- Retaining CTGAN’s residual architecture and conditional generation for robust tabular data synthesis.
- Regularizing weight distributions with KL divergence to prevent overfitting.

### Bayesian Generator

The generator maps a latent vector $z \in \mathbb{R}^{d_z}$ and a conditional vector $c \in \mathbb{R}^{d_c}$ to synthetic data $\hat{x} \in \mathbb{R}^{d_x}$. Unlike CTGAN’s deterministic generator, the Bayesian Generator treats weights as random variables with variational distributions:

$$
q(w) = \mathcal{N}(w | \mu, \sigma^2)
$$

where $\mu$ and $\log \sigma^2$ are learnable parameters, and the prior is:

$$
p(w) = \mathcal{N}(0, 1)
$$

Weights are sampled using the reparameterization trick:

$$
w = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

This is implemented in the `BayesianResidual` and `BayesianGenerator` classes:

- **Residual Layers**: Each `BayesianResidual` layer has two linear layers (`fc_mu`, `fc_logvar`) to parameterize $\mu$ and $\log \sigma^2$, followed by batch normalization, ReLU, and input concatenation.
- **Final Layer**: The `final_mu` and `final_logvar` layers parameterize the output transformation.
- **Forward Pass**: Weights are sampled in each forward pass, making the generator stochastic.

The KL divergence regularizes $q(w)$ toward $p(w)$:

$$
D_{\text{KL}}(q(w) || p(w)) = \sum_{i,j} \frac{1}{2} \left[ 1 + \log \sigma_{i,j}^2 - \mu_{i,j}^2 - e^{\log \sigma_{i,j}^2} \right]
$$

This term is computed in the `kl_divergence` methods and added to the Generator Loss with weight $\beta = 0.001$.



