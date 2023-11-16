import torch
import json
import os.path as osp
import argparse, random, sys
from types import SimpleNamespace
from dataloader import get_dataloader
from models.gan import GAN
from models.dcgan import DCGAN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Adversarial Network (GAN)',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, choices=['gan', 'dcgan', 'wgan'], required=True,
                        help='Select the model to run, where:\n'
                            '- gan: GAN\n'
                            '- dcgan: Deep Convolutional GAN\n'
                            '- wgan: Wasserstein GAN')
    eval_args = parser.add_argument_group('Evaluation mode arguments')
    eval_args.add_argument('--eval', action=argparse.BooleanOptionalAction, required=True,
                        help='Use the tag --eval to enable evaluation mode, --no-eval to enable training mode')
    eval_args.add_argument('--log-dir', type=str, required='--eval' in sys.argv,
                        help='Path to the log directory, which stores model file, config file, etc')
    training_args = parser.add_argument_group('Training mode arguments')
    training_args.add_argument('--dataroot', type=str, required='--eval' not in sys.argv,
                        help='Path to the dataset directory')
    training_args.add_argument('--dataset', type=str, choices=['mnist', 'cifar-10'], default='mnist',
                        help='Name of the dataset')
    training_args.add_argument('--exp-name', type=str,
                        help='Experiment name')
    training_args.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    training_args.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    training_args.add_argument('--steps', type=int, default=1,
                        help='Number of steps to apply to the discriminator')
    training_args.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    training_args.add_argument('--d-lr', type=float, default=2e-4,
                        help='Learning rate for optimizing the discriminator')
    training_args.add_argument('--g-lr', type=float, default=2e-4,
                        help='Learning rate for optimizing the generator')
    training_args.add_argument('--latent-dim', type=int, default=100,
                        help='Dimensionality of the latent space')
    training_args.add_argument('--sample-size', type=int, default=36,
                        help='Number of images being sampled after each epoch')
    args = parser.parse_args()

    if args.model == 'gan':
        model_name = GAN
    elif args.model == 'dcgan':
        model_name = DCGAN
    else:
        pass

    if args.eval:
        with open(osp.join(args.log_dir, 'config.json')) as f:
            config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        config.eval = args.eval
        model = model_name(config)
        model.evaluate(osp.join(args.log_dir, 'model.pt'))
        model.generate_latent_walk(args.log_dir)
    else:
        del args.log_dir
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataloader = get_dataloader(args)
        args.image_dim = list(dataloader.dataset[0][0].shape)
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        model = model_name(args)
        model.train(dataloader)
