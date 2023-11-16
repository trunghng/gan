import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid, save_image
import os, time
import os.path as osp
from typing import List
from logger import Logger
from tqdm import tqdm
import utils


class Generator(nn.Module):

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            # Input (100x1x1) -> Output (512x4x4)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Input (512x4x4) -> Output (256x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Input (256x8x8) -> Output (128x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Input (128x16x16) -> Output (channels x 32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    def forward(self, z: torch.Tensor):
        return self.network(z)


class Discriminator(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            # Input (channels x 32x32) -> Output (256x16x16)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Input (256x16x16) -> Output (512x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Input (512x8x8) -> Output (1024x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            # Input (1024x4x4) -> Output (1x1x1)
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor):
        return self.network(x)


class DCGAN:


    def __init__(self, args):
        self.device = torch.device(args.device)
        self.discriminator = Discriminator(args.image_dim[0]).to(self.device)
        self.generator = Generator(args.latent_dim, args.image_dim[0]).to(self.device)
        self.d_optim = Adam(self.discriminator.parameters(), args.d_lr, betas=(0.5, 0.999))
        self.g_optim = Adam(self.generator.parameters(), args.g_lr, betas=(0.5, 0.999))
        self.epochs = args.epochs
        self.steps = args.steps
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.image_dim = args.image_dim
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = osp.join(os.getcwd(), 'results', exp_name)
        else:
            exp_name = None
            log_dir = None
        self.logger = Logger(log_dir=log_dir, exp_name=exp_name, is_eval=args.eval)
        del args.eval
        config_dict = vars(args)
        self.logger.save_config(config_dict)
        self.logger.set_saver({
            'd_model': self.discriminator,
            'g_model': self.generator
        })


    def sample_noise(self, sample_size: int):
        return torch.randn(sample_size, self.generator.latent_dim, 1, 1)


    def train(self, dataloader):
        # Generate a fixed noise for image sampling after each epoch
        noise = self.sample_noise(self.sample_size).to(self.device)
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            for i, (imgs, _) in enumerate(tqdm(dataloader)):
                if i == len(dataloader.dataset) // self.batch_size:
                    break

                for _ in range(self.steps):
                    self.d_optim.zero_grad()
                    Z = self.sample_noise(self.batch_size).to(self.device)
                    X = imgs.to(self.device)

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
                'g_loss': g_loss.item(),
                'time': time.time() - start_time
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
        utils.save_images(fpath, samples.data.cpu(), self.image_dim)
        print('Images generated by the model are saved at', fpath)


    def generate_latent_walk(self, save_dir, steps=10):
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
