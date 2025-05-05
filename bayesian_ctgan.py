import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctgan.data_transformer import DataTransformer
from ctgan.data_sampler import DataSampler

class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


# Bayesian Residual Layer
class BayesianResidual(nn.Module):
    def __init__(self, i, o):
        super(BayesianResidual, self).__init__()
        self.fc_mu = nn.Linear(i, o)
        self.fc_logvar = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_):
        mu = self.fc_mu(input_)
        logvar = self.fc_logvar(input_)
        out = self.reparameterize(mu, logvar)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)

    def kl_divergence(self):
        kl = 0
        for param_mu, param_logvar in [(self.fc_mu.weight, self.fc_logvar.weight), (self.fc_mu.bias, self.fc_logvar.bias)]:
            kl += -0.5 * torch.sum(1 + param_logvar - param_mu.pow(2) - param_logvar.exp())
        return kl

# Bayesian Generator
class BayesianGenerator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(BayesianGenerator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [BayesianResidual(dim, item)]
            dim += item
        self.residuals = nn.ModuleList(seq)
        self.final_mu = nn.Linear(dim, data_dim)
        self.final_logvar = nn.Linear(dim, data_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_):
        x = input_
        for residual in self.residuals:
            x = residual(x)
        mu = self.final_mu(x)
        logvar = self.final_logvar(x)
        data = self.reparameterize(mu, logvar)
        return data

    def kl_divergence(self):
        kl = 0
        for residual in self.residuals:
            kl += residual.kl_divergence()
        for param_mu, param_logvar in [(self.final_mu.weight, self.final_logvar.weight), (self.final_mu.bias, self.final_logvar.bias)]:
            kl += -0.5 * torch.sum(1 + param_logvar - param_mu.pow(2) - param_logvar.exp())
        return kl

class BayesianCTGAN:

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    # Apply Activate (unchanged)
    def _apply_activate(self, data, transformer):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = F.gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    # Cond Loss (unchanged)
    def _cond_loss(self, data, c, m, transformer):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c
        loss = torch.stack(loss, dim=1)
        return (loss * m).sum() / data.size()[0]

    def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def fit(self, train_data, discrete_columns=(), epochs=None):

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions
       
        generator = BayesianGenerator(self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim).to(self._device)
        discriminator = Discriminator(data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim).to(self._device)

        optimizerG = torch.optim.Adam(
                    generator.parameters(),
                    lr=2e-4,
                    betas=(0.5, 0.9),
                    weight_decay=1e-6
                )

        optimizerD = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=2e-4,
                    betas=(0.5, 0.9),
                    weight_decay=1e-6
                )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        # Training Loop
        num_epochs = 300
        discriminator_steps = 1
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        kl_weight = 0.005  # Added for Bayesian regularization


        for i in range(num_epochs):
            for id_ in range(steps_per_epoch):
                # Discriminator Training
                for n in range(discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std).to(self._device)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                    fake = generator(fakez)
                    fakeact = self._apply_activate(fake, transformer=self._transformer)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, pac=10)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # Generator Training
                fakez = torch.normal(mean=mean, std=std).to(self._device)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = generator(fakez)
                fakeact = self._apply_activate(fake, transformer=self._transformer)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1, transformer=self._transformer)

                kl_div = generator.kl_divergence()  # Bayesian KL divergence term
                loss_g = -torch.mean(y_fake) + cross_entropy + kl_weight * kl_div

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            kl_div_value = kl_div.detach().cpu().item()

            print(f"Epoch [{i+1}/{num_epochs}], D Loss: {discriminator_loss:.4f}, G Loss: {generator_loss:.4f}, KL Div: {kl_div_value:.4f}")

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            with torch.no_grad():
                outputs = [self._generator(fakez).cpu().numpy() for _ in range(10)]
                variance = np.var(outputs, axis=0)
                print(f"Output variance: {variance.mean():.4f}")
                
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake, transformer=self._transformer)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)
