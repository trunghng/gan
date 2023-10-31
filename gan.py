import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


class Generator(nn.Module):

    def __init__(self, input_dim=100):
        super().__init__()
        self.noise_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 32*32),
            nn.Tanh()
        )


    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.network(x)


class GAN:


    def __init__(self,
                discriminator,
                generator,
                device,
                epochs=20,
                steps=1,
                batch_size=64,
                d_lr=2e-4,
                g_lr=2e-4):
        self.discriminator = discriminator
        self.generator = generator
        self.device = device
        self.d_optim = Adam(self.discriminator.parameters(), d_lr, betas=(0.5, 0.999))
        self.g_optim = Adam(self.generator.parameters(), g_lr, betas=(0.5, 0.999))
        self.epochs = epochs
        self.steps = steps
        self.batch_size = batch_size


    def sample_noise(self, batch_size):
        return torch.rand(batch_size, self.generator.noise_dim)


    def train(self, dataloader):
        d_losses, g_losses = [], []

        for epoch in range(1, self.epochs + 1):
            for i, (imgs, _) in enumerate(dataloader):
                if i == len(dataloader.dataset) // self.batch_size:
                    break

                for _ in range(self.steps):
                    self.d_optim.zero_grad()
                    Z = self.sample_noise(self.batch_size).to(self.device)
                    X = imgs.reshape(self.batch_size, -1).to(self.device)

                    d_loss_real = nn.BCELoss()(self.discriminator(X).reshape(self.batch_size), torch.ones(self.batch_size, device=self.device))
                    d_loss_fake = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(self.batch_size), torch.zeros(self.batch_size, device=self.device))
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.d_optim.step()

                self.g_optim.zero_grad()
                Z = self.sample_noise(self.batch_size).to(device)
                g_loss = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(self.batch_size), torch.ones(self.batch_size, device=self.device))
                g_loss.backward()
                self.g_optim.step()
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

                if ((i + 1) % 100) == 0:
                    print("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, i + 1, \
                        len(dataloader.dataset) // self.batch_size, d_loss.item(), g_loss.item()))
                    plot(self.generator(Z)[:25].data.cpu().numpy(), f'epoch{epoch}_{i + 1}.png')
        return {'d_loss': d_losses, 'g_loss': g_losses}


def plot(X, fname):
    for i in range(36):
        plt.subplot(6, 6, 1 + i)
        plt.axis('off')
        plt.imshow(X[i].reshape(32, 32), cmap='gray')
    plt.savefig(fname)


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)

    data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
    )

    dataloader = DataLoader(data, batch_size=64, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    gan = GAN(discriminator, generator, device)
    gan.train(dataloader)
