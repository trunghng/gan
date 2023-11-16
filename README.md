# GAN
Pytorch implementation of Generative Adversarial Network (GAN) models:
- (Original) GAN
- Deep Convolutional GAN
- Wasserstein GAN

## Getting started

1. Clone the repository
```bash
git clone https://github.com/trunghng/gan <project_name>
```

2. Install dependencies
```bash
cd <project_name>
pip install -r requirements.txt
```

## Usage
```bash
usage: run.py [-h] --model {gan,dcgan,wgan} --eval | --no-eval [--log-dir LOG_DIR] --dataroot
              DATAROOT [--dataset {mnist,cifar-10}] [--exp-name EXP_NAME] [--seed SEED]
              [--epochs EPOCHS] [--steps STEPS] [--batch-size BATCH_SIZE] [--d-lr D_LR]
              [--g-lr G_LR] [--latent-dim LATENT_DIM] [--sample-size SAMPLE_SIZE]

Generative Adversarial Network (GAN)

optional arguments:
  -h, --help            show this help message and exit
  --model {gan,dcgan,wgan}
                        Select the model to run, where:
                        - gan: GAN
                        - dcgan: Deep Convolutional GAN
                        - wgan: Wasserstein GAN

Evaluation mode arguments:
  --eval, --no-eval     Use the tag --eval to enable evaluation mode, --no-eval to enable training mode
  --log-dir LOG_DIR     Path to the log directory, which stores model file, config file, etc

Training mode arguments:
  --dataroot DATAROOT   Path to the dataset directory
  --dataset {mnist,cifar-10}
                        Name of the dataset
  --exp-name EXP_NAME   Experiment name
  --seed SEED           Seed for RNG
  --epochs EPOCHS       Number of epochs
  --steps STEPS         Number of steps to apply to the discriminator
  --batch-size BATCH_SIZE
                        Minibatch size
  --d-lr D_LR           Learning rate for optimizing the discriminator
  --g-lr G_LR           Learning rate for optimizing the generator
  --latent-dim LATENT_DIM
                        Dimensionality of the latent space
  --sample-size SAMPLE_SIZE
                        Number of images being sampled after each epoch
```

### Training
For model training, for instance, in order to train GAN with default settings, we simply run
```bash
python run.py --model gan --no-eval --dataroot data/mnist
```
which is equivalent to
```bash
python run.py --model gan --no-eval --dataroot data/mnist --dataset mnist \
--exp-name gan-mnist --seed 0 --epochs 50 --steps 1 --batch-size 64 \
--d-lr 0.0002 --g-lr 0.0002 --latent-dim 100 --sample-size 36
```

### Evaluation
For model evaluation, for example, we evaluate GAN with log files saved at ```results/gan-mnist``` as
```bash
python run.py --model gan --eval --log-dir results/gan-mnist
```

## References
[1] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. [Generative Adversarial Nets](http://papers.neurips.cc/paper/5423-generative-adversarial-nets.pdf). NIPS, 2014.  
[2] Alec Radford, Luke Metz, Soumith Chintala. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). arXiv preprint, arXiv:1511.06434, 2016.  
[3] Martin Arjovsky, Soumith Chintala, Léon Bottou. [Wasserstein GAN](https://arxiv.org/abs/1701.07875). arXiv preprint, arXiv:1701.07875, 2017.


## License
[MIT](https://choosealicense.com/licenses/mit/)