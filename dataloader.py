from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os.path as osp


def get_dataloader(args):
    download = True

    if args.dataset == 'mnist':
        if osp.exists(osp.join(args.dataroot, 'MNIST')):
            download = False

        data = datasets.MNIST(
            root=args.dataroot,
            train=True,
            download=download,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        )

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'cifar-10':
        pass

    return dataloader