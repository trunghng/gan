import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid, save_image
import argparse, random, os, json, sys
import os.path as osp
from typing import List
from types import SimpleNamespace
from dataloader import get_dataloader
from logger import Logger
import utils


class Generator(nn.Module):

    def __init__(self, latent_dim: int, output_dim: List[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(output_dim)),
            nn.Tanh()
        )


    def forward(self, z: torch.Tensor):
        return self.network(z)


class Discriminator(nn.Module):

    def __init__(self, input_dim: List[int]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(input_dim), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor):
        return self.network(x)


class GAN:


    def __init__(self, device, args):
        self.discriminator = Discriminator(args.image_size).to(device)
        self.generator = Generator(args.latent_dim, args.image_size).to(device)
        self.device = device
        self.d_optim = Adam(self.discriminator.parameters(), args.d_lr, betas=(0.5, 0.999))
        self.g_optim = Adam(self.generator.parameters(), args.g_lr, betas=(0.5, 0.999))
        self.epochs = args.epochs
        self.steps = args.steps
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.image_size = args.image_size
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'results', exp_name, f'{exp_name}_s{args.seed}')
        else:
            exp_name = None
            log_dir = None
        self.logger = Logger(log_dir=log_dir, exp_name=exp_name)
        config_dict = vars(args)
        config_dict['model'] = 'gan'
        self.logger.save_config(config_dict)
        self.logger.set_saver({
            'd_model': self.discriminator,
            'g_model': self.generator
        })


    def sample_noise(self, sample_size: int):
        return torch.randn(sample_size, self.generator.latent_dim)


    def train(self, dataloader):
        # Generate a fixed noise for image sampling after training
        noise = self.sample_noise(self.sample_size).to(self.device)

        for epoch in range(1, self.epochs + 1):
            for i, (imgs, _) in enumerate(dataloader):
                if i == len(dataloader.dataset) // self.batch_size:
                    break

                for _ in range(self.steps):
                    self.d_optim.zero_grad()
                    Z = self.sample_noise(self.batch_size).to(self.device)
                    X = imgs.reshape(self.batch_size, -1).to(self.device)

                    d_loss_real = nn.BCELoss()(self.discriminator(X).reshape(self.batch_size), 
                                                torch.ones(self.batch_size, device=self.device))
                    d_loss_fake = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(self.batch_size), 
                                                torch.zeros(self.batch_size, device=self.device))
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.d_optim.step()

                self.g_optim.zero_grad()
                Z = self.sample_noise(self.batch_size).to(self.device)
                g_loss = nn.BCELoss()(self.discriminator(self.generator(Z)).reshape(self.batch_size), 
                                        torch.ones(self.batch_size, device=self.device))
                g_loss.backward()
                self.g_optim.step()

            self.logger.log({
                'epoch': epoch,
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item()
            })

            # Generate images and denormalize
            samples = self.generator(noise).mul(0.5).add(0.5)
            self.logger.generate_imgs(samples.data.cpu(), epoch)
        self.logger.generate_gif()
        self.logger.plot()
        self.logger.save_model()


    def load_model(self, model_path: str):
        model = torch.load(model_path)
        self.discriminator.load_state_dict(model['d_model'])
        self.generator.load_state_dict(model['g_model'])


    def evaluate(self, model_path: str):
        self.load_model(model_path)
        noise = self.sample_noise(self.sample_size).to(self.device)
        samples = self.generator(noise).mul(0.5).add(0.5)
        fpath = osp.join(osp.dirname(model_path), f'test.png')
        utils.save_images(fpath, samples.data.cpu(), self.image_size)
        print('Images generated by the model are saved at', fpath)


    def generate_latent_walk(self, steps, save_dir):
        z1 = self.sample_noise(1).to(self.device)
        z2 = self.sample_noise(1).to(self.device)
        step_size = 1.0 / steps
        images = []
        for i in range(steps + 1):
            z = z1 * i * step_size + z2 * (1 - i * step_size)
            image = self.generator(z).mul(0.5).add(0.5)
            images.append(image.view(-1, 32, 32).data.cpu())
        fpath = osp.join(save_dir, 'latent_walk.png')
        img = make_grid(images, nrow=steps + 1)
        save_image(img, fpath)
        print('Latent walk generated by the model is saved at', fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Adversarial Network (GAN)')
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction,
                        help='Use the tag to enable evaluation mode')
    parser.add_argument('--log-dir', type=str, required='--eval' in sys.argv,
                        help='Path to the log directory, which stores model file, config file, etc')
    parser.add_argument('--dataroot', type=str, required='--eval' not in sys.argv,
                        help='Path to dataset folder')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar-10'], default='mnist',
                        help='Name of the dataset')
    parser.add_argument('--exp-name', type=str, default='gan-mnist',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of steps to apply to the discriminator')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--d-lr', type=float, default=2e-4,
                        help='Learning rate for optimizing the discriminator')
    parser.add_argument('--g-lr', type=float, default=2e-4,
                        help='Learning rate for optimizing the generator')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='Dimensionality of the latent space')
    parser.add_argument('--sample-size', type=int, default=36,
                        help='Number of images being sampled after each epoch')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval:
        with open(osp.join(args.log_dir, 'config.json')) as f:
            config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
            model = GAN(device, config)
            model.evaluate(osp.join(args.log_dir, 'model.pt'))
            model.generate_latent_walk(10, args.log_dir)
    else:
        del args.eval, args.log_dir
        dataloader = get_dataloader(args)
        args.image_size = list(dataloader.dataset[0][0].shape)

        model = GAN(device, args)
        model.train(dataloader)
