import os.path as osp
import os, atexit, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from math import ceil, log10
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from imageio import imread, mimsave


class Logger:

    def __init__(self, 
                log_dir=None,
                log_fname='progress.txt',
                exp_name=None):
        self.exp_name = exp_name
        self.log_dir = log_dir if log_dir else f'/tmp/experiments/{str(datetime.now())}'
        self.imgs_dir = osp.join(self.log_dir, 'imgs')
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(self.imgs_dir)
        self.log_file = open(osp.join(self.log_dir, log_fname), 'w')
        atexit.register(self.log_file.close)
        self.first_row = True
        self.record = defaultdict(list)


    def set_saver(self, model):
        self.model = model


    def save_config(self, config):
        if self.exp_name is not None:
            config['exp_name'] = self.exp_name
        output = json.dumps(config, separators=(',',':\t'), indent=4)
        print('Experiment config:\n', output)
        with open(osp.join(self.log_dir, 'config.json'), 'w') as out:
            out.write(output)
        self.config = config


    def log(self, data):
        assert self.log_file is not None, "Logging output file name must be defined."
        if self.first_row:
            self.log_file.write("\t".join(data.keys()) + "\n")
        values = []
        logstr = []
        for key in data:
            value = data.get(key)
            valstr = "%7.3g" % value if hasattr(value, "__float__") else value
            logstr.append(f'{key} {valstr}')
            values.append(value)
            self.record[key].append(value)
        print(' | '.join(logstr))
        self.log_file.write("\t".join(map(str, values)) + "\n")
        self.log_file.flush()
        self.first_row = False


    def generate_imgs(self, X, epoch, nrow=6):
        n = ceil(log10(self.config['epochs']))
        img = make_grid(X.reshape([X.shape[0], 1, 32, 32]), nrow=nrow)
        save_image(img, osp.join(self.imgs_dir, f'ep_{str(epoch).zfill(n)}.png'))


    def generate_gif(self):
        images = list()
        for file in sorted(Path(self.imgs_dir).iterdir()):
            if not file.is_file():
                continue
            images.append(imread(file))
        mimsave(osp.join(self.log_dir, 'samples.gif'), images, fps=5)


    def plot(self):
        ax = plt.figure().gca()
        ax.plot(self.record['epoch'], self.record['d_loss'], label='Discriminator loss')
        ax.plot(self.record['epoch'], self.record['g_loss'], label='Generator loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper left')
        plt.savefig(osp.join(self.log_dir, 'loss.png'))


    def save_model(self):
        path = osp.join(self.log_dir, 'model.pt')
        torch.save({
            'd_model': self.model['d_model'].state_dict(),
            'g_model': self.model['g_model'].state_dict()
        }, path)
        print(f'Model is saved successfully at {path}')
