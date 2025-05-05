Bayesian Conditional Tabular GAN (Bayesian CTGAN)
Overview
This repository implements a Bayesian Conditional Tabular GAN (Bayesian CTGAN), a novel extension of the Conditional Tabular GAN (CTGAN) for generating synthetic tabular data. The key innovation is the integration of a Bayesian Generator with a residual architecture, which models weights as random variables to enhance output diversity and enable uncertainty quantification. The model uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) framework, augmented with conditional generation for mixed continuous and discrete columns, and a PacGAN approach to prevent mode collapse.
The implementation is designed for tabular datasets with mixed data types, addressing privacy-preserving data sharing in domains like finance and healthcare. This README explains the theoretical foundations, including the learning of weight distributions, the Discriminator and Generator losses, and provides instructions for running the code.
Theoretical Foundations
Motivation
Synthetic tabular data generation is critical for privacy-preserving data sharing and augmentation. CTGAN [Xu et al., 2019] is a state-of-the-art model that uses WGAN-GP and conditional generation to handle mixed data types. However, its deterministic generator limits output diversity and lacks uncertainty quantification. The Bayesian CTGAN addresses these limitations by:

Modeling generator weights as random variables using variational inference, increasing diversity and enabling uncertainty estimates.
Retaining CTGAN’s residual architecture and conditional generation for robust tabular data synthesis.
Regularizing weight distributions with KL divergence to prevent overfitting.

Bayesian Generator
The generator maps a latent vector $( z \in \mathbb{R}^{d_z} )$ and a conditional vector ( c \in \mathbb{R}^{d_c} ) to synthetic data ( \hat{x} \in \mathbb{R}^{d_x} ). Unlike CTGAN’s deterministic generator, the Bayesian Generator treats weights as random variables with variational distributions:[q(w) = \mathcal{N}(w | \mu, \sigma^2)]where (\mu) and (\log \sigma^2) are learnable parameters, and the prior is:[p(w) = \mathcal{N}(0, 1)]Weights are sampled using the reparameterization trick:[w = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)]This is implemented in the BayesianResidual and BayesianGenerator classes:

Residual Layers: Each BayesianResidual layer has two linear layers (fc_mu, fc_logvar) to parameterize (\mu) and (\log \sigma^2), followed by batch normalization, ReLU, and input concatenation.
Final Layer: The final_mu and final_logvar layers parameterize the output transformation.
Forward Pass: Weights are sampled in each forward pass, making the generator stochastic.

The KL divergence regularizes ( q(w) ) toward ( p(w) ):[D_{\text{KL}}(q(w) || p(w)) = \sum_{i,j} \frac{1}{2} \left[ 1 + \log \sigma_{i,j}^2 - \mu_{i,j}^2 - e^{\log \sigma_{i,j}^2} \right]]This term is computed in the kl_divergence methods and added to the Generator Loss with weight (\beta = 0.001).
Discriminator and Generator Losses
The model uses WGAN-GP [Gulrajani et al., 2017] with a PacGAN approach [Lin et al., 2018] to train the critic (discriminator) and generator adversarially.
Discriminator Loss
The Discriminator Loss measures the critic’s ability to distinguish real data from fake data, while enforcing a 1-Lipschitz constraint:[L_D = - \left( \mathbb{E}{x \sim P_r}[f(x, c)] - \mathbb{E}{z \sim P_z, w \sim q(w)}[f(G(z, c; w), c)] \right) + \lambda \cdot \text{GP}]

( f(\cdot, c) ): Critic’s score for data (real or fake) and conditional vector ( c ).
( \lambda = 10 ): Gradient penalty coefficient.
( \text{GP} ): Gradient penalty, computed over interpolated samples:[\text{GP} = \mathbb{E}{\hat{x} \sim P{\hat{x}}} \left[ \left( |\nabla_{\hat{x}} f(\hat{x}, c)|_2 - 1 \right)^2 \right]]
Code: Implemented in the training loop:y_fake = discriminator(fake_cat)
y_real = discriminator(real_cat)
pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, device, pac=10)
loss_d = -(torch.mean(y_real) - torch.mean(y_fake))


Role: Negative ( L_D ) (e.g., -0.6195) indicates the critic assigns higher scores to real data. Small magnitudes reflect a balanced dynamic, aided by PacGAN (pac=10).

Generator Loss
The Generator Loss encourages the generator to fool the critic, match discrete column distributions, and regularize weight distributions:[L_G = -\mathbb{E}{z \sim P_z, w \sim q(w)}[f(G(z, c; w), c)] + L{\text{cond}} + \beta \cdot D_{\text{KL}}(q(w) || p(w))]

Adversarial Term: Maximizes critic scores for fake data.
Conditional Loss (( L_{\text{cond}} )): Cross-entropy loss ensuring discrete columns match the conditional vector:[L_{\text{cond}} = \frac{1}{N} \sum_{i=1}^N \sum_{j \in \text{discrete}} m_{i,j} \cdot \text{CE}(G(z_i, c_i; w)j, c{i,j})]
KL Divergence: Regularizes weight distributions, with (\beta = 0.001).
Code: Implemented in the training loop:y_fake = discriminator(torch.cat([fakeact, c1], dim=1)) if c1 is not None else discriminator(fakeact)
cross_entropy = cond_loss(fake, c1, m1, transformer=transformer) if condvec is not None else 0
kl_div = generator.kl_divergence()
loss_g = -torch.mean(y_fake) + cross_entropy + kl_weight * kl_div


Role: Negative ( L_G ) (e.g., -0.2334) indicates the adversarial term dominates, showing generator improvement. Positive ( L_G ) (e.g., 0.4706) reflects large ( L_{\text{cond}} ).



The shift to negative G Loss and stable D Loss reflect generator improvement, with ( L_{\text{cond}} ) likely dominating positive G Loss values. The decreasing KL Div confirms effective weight distribution learning.
Installation



References

Xu, L., et al. (2019). Modeling Tabular Data using Conditional GAN. NeurIPS.
Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. NeurIPS.
Lin, Z., et al. (2018). PacGAN: The Power of Two Samples in Generative Adversarial Networks. NeurIPS.
Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks. ICML.
Saatci, Y., & Wilson, A. G. (2017). Bayesian GAN. NeurIPS.

License
MIT License. See LICENSE for details.
