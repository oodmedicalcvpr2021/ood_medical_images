import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OODDataset
import utils
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from network import WideResNet
from utils import plot_devries_histograms
import dataset
from dataset import Cutout
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
import torchvision


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--idd_name", default="retina")
    parser.add_argument("--mode", type=str, default='devries',
                        choices=['baseline', 'devries', 'oe'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['skeletal-age', 'mura', 'mimic-crx'])
    parser.add_argument("--network", type=str, default="WideResNet")
    parser.add_argument("--ckpt", type=str, default="")

    # Misc
    parser.add_argument("--load_memory", type=bool, default=False, help="Load images into CPU")
    args = parser.parse_args()
    args = utils.compute_args(args)

    dataset.test_input_transforms.transforms[0] = Resize(size=(32, 32))
    dataset.train_input_transforms = Compose([
        RandomResizedCrop((32, 32)),
        RandomHorizontalFlip(),
        ToTensor(),
        Cutout(16),
    ])

    # Create dataloader according to experiments
    loader_args = {'name': args.idd_name,
                   'root_dir': args.root[args.idd_name],
                   'csv_file': 'btrain.csv',
                   'load_memory': args.load_memory}

    loaders = dict()
    loaders[args.idd_name] = DataLoader(OODDataset(**dict(loader_args, **{'csv_file': 'btest.csv',
                                                                          'mode': 'test'})),
                                        batch_size=8,
                                        shuffle=False,
                                        num_workers=4)

    for ood_name in args.ood_name:
        loaders[ood_name] = DataLoader(OODDataset(**{'name': ood_name,
                                                     'mode': 'test',
                                                     'root_dir': args.root[ood_name],
                                                     'csv_file': 'btest.csv',
                                                     'load_memory': False
                                                     }),
                                       batch_size=8,
                                       shuffle=False,
                                       num_workers=4)

    loaders["SVHN"] = DataLoader(torchvision.datasets.SVHN(
        'SVHN', split='test', download=True,
        transform=dataset.test_input_transforms),
        batch_size=16,
        shuffle=True,
        num_workers=4)

    loaders["cifar100"] = DataLoader(torchvision.datasets.CIFAR100(
        'cifar', train=False, download=True,
        transform=dataset.train_input_transforms),
        batch_size=16,
        shuffle=True,
        num_workers=4)


    net = eval(args.network)(num_classes=args.num_classes).cuda()

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt["net"])
    net.eval()

    print("Accuracy", ckpt["accuracy"])
    print("OOD Metrics", '\n'.join([str(v) for v in ckpt["ood_metrics"]]))

    confs = dict()
    # Get lowest and highest confidence for each loaders
    for dataset_name, test_false_loader in loaders.items():
        confs[dataset_name] = [0.0, None, 1.0, None, []]
        for train_iter, sample in enumerate(test_false_loader, 0):
            data, labels = sample[0].cuda(), sample[1].cuda()
            pred, confidence = net(data.cuda())
            confidence = torch.sigmoid(confidence).cpu().data
            confs[dataset_name][4].extend(confidence)
            for i, c in enumerate(confidence):
                if c.cpu().item() > confs[dataset_name][0]:
                    confs[dataset_name][0] = c.cpu().item()
                    confs[dataset_name][1] = data[i]
                if c.cpu().item() < confs[dataset_name][2]:
                    confs[dataset_name][2] = c.cpu().item()
                    confs[dataset_name][3] = data[i]

    ckpt_dir = os.path.dirname(args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Dumping outputs in", ckpt_dir)

    # Saving images for lowest and highest confidence for each loader
    for dataset_name, values in confs.items():
        print(dataset_name, "max conf", round(values[0], 2), "min conf", round(values[2], 2))

        t = values[1].permute(1, 2, 0).cpu().data.numpy()
        t = (t * 255 / np.max(t)).astype('uint8')
        x = Image.fromarray(t)
        x.save(os.path.join(ckpt_dir, str(dataset_name) + "_" + str(round(values[0], 2)) + ".jpg"), "jpeg")

        t = values[3].permute(1, 2, 0).cpu().data.numpy()
        t = (t * 255 / np.max(t)).astype('uint8')
        x = Image.fromarray(t)
        x.save(os.path.join(ckpt_dir, str(dataset_name) + "_" + str(round(values[2], 2)) + ".jpg"), "jpeg")

    # Computing confidence histrograms
    idd = confs.pop(args.idd_name)
    for ood_name, values in confs.items():
        plot_devries_histograms(ind_scores=np.array(idd[4]),
                                ood_scores=np.array(values[4]),
                                name=args.idd_name + ' vs ' + ood_name) \
            .savefig(os.path.join(ckpt_dir, ood_name + "histogram.jpg"))


if __name__ == "__main__":
    main()
