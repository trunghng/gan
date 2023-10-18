import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tensorflow.python.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt


def mlp(dims,
        activation,
        output_activation=nn.Sigmoid):
    layers = []
    for i in range(len(dims) - 1):
        activation_ = activation if i < len(dims) - 2 else output_activation
        layers += [nn.Linear(dims[i], dims[i + 1]), activation_()]
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self,
                input_dim=28*28,
                hidden_dims=[240, 240],
                output_dim=1,
                activation=nn.Sigmoid):
        super().__init__()
        self.network = mlp([input_dim, *hidden_dims, output_dim], nn.ReLU, activation)


    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):

    def __init__(self,
                input_dim=64,
                hidden_dims=[1200, 1200],
                output_dim=28*28,
                activation=nn.Tanh):
        super().__init__()
        self.noise_dim = input_dim
        self.network = mlp([input_dim, *hidden_dims, output_dim], nn.ReLU, activation)


    def forward(self, z):
        return self.network(z)


class GAN:


    def __init__(self,
                discriminator,
                generator,
                training_data,
                d_optim=Adam,
                g_optim=Adam,
                d_lr=1e-3,
                g_lr=1e-3):
        self.discriminator = discriminator
        self.generator = generator
        self.training_data = training_data
        self.d_optim = d_optim(self.discriminator.parameters(), d_lr)
        self.g_optim = g_optim(self.generator.parameters(), g_lr)


    def sample_example(self, batch_size):
        # indices = torch.randperm(batch_size)
        # return self.training_data[indices]
        indices = torch.randperm(self.training_data.shape[0])[:batch_size]
        return torch.tensor(self.training_data[indices], dtype=torch.float).reshape(batch_size, -1)


    def sample_noise(self, batch_size):
        return torch.rand([batch_size, self.generator.noise_dim])


    def train(self, n_iters=50000, n_steps=1, batch_size=100):
        d_losses, g_losses = [], []

        for i in range(n_iters):
            for _ in range(n_steps):
                Z = self.sample_noise(batch_size)
                X = self.sample_example(batch_size)
                # d_loss = (np.log(self.discriminator(X)) + np.log(1 - self.discriminator(self.generator(Z)))).mean()
                d_loss_real = nn.BCELoss()(self.discriminator(X).reshape(batch_size), torch.ones(batch_size))
                d_loss_fake = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(batch_size), torch.zeros(batch_size))
                d_loss = d_loss_real + d_loss_fake
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

            Z = self.sample_noise(batch_size)
            # g_loss = np.log(1 - self.discriminator(self.generator(Z))).mean()
            g_loss = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(batch_size), torch.zeros(batch_size))
            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            if (i + 1) % 100 == 0:
                print(f'iter {i+1}: d_loss={d_loss}, g_loss={g_loss}') 
        return {'d_loss': d_losses, 'g_loss': g_losses}


def plot(X):
    for i in range(25):
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(X[i], cmap='gray')
    plt.savefig("MNIST_data.png")
    plt.show()
    

if __name__ == '__main__':
    (x_train, _), (_, _) = load_data()
    # plot(x_train[0:25, :])

    # normalize training data from [0, 255] to [-1, 1]
    x_train = (np.float32(x_train) - 127.5) / 127.5

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator()
    generator = Generator()
    gan = GAN(discriminator, generator, x_train)
    gan.train()
    plot(gan.generator(gan.sample_noise(25)))














