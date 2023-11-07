import torch
import torch.nn as nn
from torch.optim import Adam
import argparse, random, os
from dataloader import get_dataloader
from logger import Logger


class Generator(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32),
            nn.Tanh()
        )


    def forward(self, z: torch.Tensor):
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


    def forward(self, x: torch.Tensor):
        return self.network(x)


class GAN:


    def __init__(self,
                discriminator: Discriminator,
                generator: Generator,
                device: str,
                args):
        self.discriminator = discriminator
        self.generator = generator
        self.device = device
        self.d_optim = Adam(self.discriminator.parameters(), args.d_lr, betas=(0.5, 0.999))
        self.g_optim = Adam(self.generator.parameters(), args.g_lr, betas=(0.5, 0.999))
        self.epochs = args.epochs
        self.steps = args.steps
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
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
            'd_model': discriminator,
            'g_model': generator
        })


    def sample_noise(self, sample_size: int):
        return torch.randn(sample_size, self.generator.latent_dim)


    def train(self, dataloader):
        # Generate a fixed noise for image sampling after training
        noise = self.sample_noise(self.sample_size)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Adversarial Network')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to dataset folder')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar-10'], default='mnist',
                        help='Name of dataset')
    parser.add_argument('--exp-name', type=str, default='mnist',
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
                        help='Number of images sampled after training')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataloader = get_dataloader(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator().to(device)
    generator = Generator(args.latent_dim).to(device)
    gan = GAN(discriminator, generator, device, args)
    gan.train(dataloader)
