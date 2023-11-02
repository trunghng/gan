import os.path as osp
import os, atexit, json
from datetime import datetime
import matplotlib.pyplot as plt


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
        logstr = ''
        for key in data:
            value = data.get(key, "")
            valstr = "%8.4g" % value if hasattr(value, "__float__") else value
            logstr += f'[{key} {valstr}]'
            values.append(value)
        print(logstr)
        self.log_file.write("\t".join(map(str, values)) + "\n")
        self.log_file.flush()
        self.first_row = False


    def plot(self, X, epoch, batch, n_cols=6, n_rows=6):
        X_ = X[:n_cols * n_rows]
        for i in range(n_cols * n_rows):
            plt.subplot(n_cols, n_rows, i + 1)
            plt.axis('off')
            plt.imshow(X_[i].reshape(32, 32), cmap='gray')
        fname = f'ep{epoch}_b{batch}.png'
        plt.savefig(osp.join(self.imgs_dir, fname))

