import os
import time
import argparse
import dataset
import utils
import torch
import numpy as np
from metrics import calc_metrics
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OODDataset
import torch.nn as nn
import torch.optim as optim
from network import WideResNet
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
import torchvision.datasets
from dataset import Cutout
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--idd_name", default="retina")
    parser.add_argument("--mode", type=str, default='devries',
                        choices=['baseline', 'devries', 'oe'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['skeletal-age', 'mura', 'mimic-crx'])
    parser.add_argument("--network", type=str, default="WideResNet")
    # Hyper params
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)

    # Training params
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--early_stop_metric', type=str, default="fpr_at_95_tpr")
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--eval_start', type=int, default=1)

    # Misc
    parser.add_argument("--load_memory", type=bool, default=False, help="Load images into CPU")

    args = parser.parse_args()
    args = utils.compute_args(args)

    # Change to 32x32
    dataset.test_input_transforms.transforms[0] = Resize(size=(32, 32))
    dataset.train_input_transforms = Compose([
        RandomResizedCrop((32, 32)),
        RandomHorizontalFlip(),
        ToTensor(),
        Cutout(16),
    ])

    # Create dataloader according to experiments
    loader_args = {'name': args.idd_name,
                   'mode': 'train',
                   'root_dir': args.root[args.idd_name],
                   'csv_file': 'btrain.csv',
                   'load_memory': args.load_memory}

    train_loader = DataLoader(OODDataset(**loader_args),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_true_loader = DataLoader(OODDataset(**dict(loader_args, **{'csv_file': 'btest.csv', 'mode': 'test'})),
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4)

    test_false_loaders = {}
    for ood_name in args.ood_name:
        test_false_loaders[ood_name] = DataLoader(OODDataset(**{'name': ood_name,
                                                                'mode': 'test',
                                                                'root_dir': args.root[ood_name],
                                                                'csv_file': 'btest.csv',
                                                                'load_memory': False
                                                                }),
                                                  batch_size=16,
                                                  shuffle=False,
                                                  num_workers=4)

    outlier_datasets = list()
    outlier_datasets.append(torchvision.datasets.CIFAR10(
        'cifar', train=True, download=True,
        transform=dataset.train_input_transforms))

    outlier_datasets.append(torchvision.datasets.CIFAR100(
        'cifar', train=True, download=True,
        transform=dataset.train_input_transforms))

    outlier_datasets.append(torchvision.datasets.SVHN(
        'SVHN', split='test', download=True,
        transform=dataset.train_input_transforms))

    outlier_datasets.append(torchvision.datasets.ImageFolder("data/val_imagenet", transform=dataset.train_input_transforms))
    outlier_set = ConcatDataset(outlier_datasets)

    net = eval(args.network)(num_classes=args.num_classes).cuda()

    optimizer = torch.optim.SGD(
        net.parameters(), args.lr, momentum=0.9,
        weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=0.8)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

    print("Network", args.network,
          "\nTotal number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

    checkpoints_folder = f"checkpoints/{args.experiment_name}"
    os.makedirs(checkpoints_folder, exist_ok=True)
    early_stop = 0
    init_epoch = 0
    best_early_stop_value = 1.0

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        train_start = time.time()
        # Train phase
        net.train()
        # Shuffling outlier set
        outlier_loader = DataLoader(outlier_set,
                                    batch_size=16,
                                    shuffle=True,
                                    num_workers=4)

        for train_iter, (in_set, out_set) in enumerate(zip(train_loader, outlier_loader)):
            optimizer.zero_grad()

            data = torch.cat((in_set[0], out_set[0]), dim=0)
            # forward
            _, confidence = net(data.cuda())
            # backward
            labels = torch.from_numpy(np.concatenate(
                [np.ones((len(in_set[0]), 1)),
                 np.zeros((len(out_set[0]), 1))
                 ]))
            task_loss = F.binary_cross_entropy_with_logits(confidence, labels.cuda(), reduction="sum")
            task_loss.backward()
            optimizer.step()

            print(
                "\r[Epoch {}][Step {}/{}] task_loss: {:.5f}, Lr: {:.2e}, ES: {}, {:.2f} m remaining".format(
                    epoch + 1,
                    train_iter,
                    int(len(train_loader.dataset) / args.batch_size),
                    task_loss.cpu().data.numpy(),
                    *[group['lr'] for group in optimizer.param_groups],
                    early_stop,
                    ((time.time() - train_start) / (train_iter + 1)) * (
                            (len(train_loader.dataset) / args.batch_size) - train_iter) / 60,
                ), end='          ')

        # Eval phase
        if epoch + 1 >= args.eval_start:
            net.eval()

            def evaluate(data_loader):
                confidences = []

                for test_iter, sample in enumerate(data_loader, 0):
                    images, labels = sample
                    _, confidence = net(images.cuda())
                    confidence = torch.sigmoid(confidence).cpu().data
                    confidences.append(confidence)
                confidences = np.concatenate(confidences)

                return None, confidences

            # In domain evaluation
            _, ind_confidences = evaluate(test_true_loader)
            ind_labels = np.ones(ind_confidences.shape[0])

            # accuracy = round(float(np.mean(ind_predictions)), 4)
            # print(args.idd_name, accuracy, '% accuracy')

            # Out of domain evaluation
            early_stop_metric_value = 0
            ood_metric_dicts = []
            for ood_name, test_false_loader in test_false_loaders.items():
                _, ood_confidences = evaluate(test_false_loader)
                ood_labels = np.zeros(ood_confidences.shape[0])

                labels = np.concatenate([ind_labels, ood_labels])
                scores = np.concatenate([ind_confidences, ood_confidences])

                ood_metrics = calc_metrics(scores, labels)
                ood_metrics['OOD Name'] = ood_name
                print(str(ood_metrics))
                ood_metric_dicts.append(ood_metrics)
                early_stop_metric_value += ood_metrics[args.early_stop_metric]

            early_stop_metric_value = early_stop_metric_value / len(test_false_loaders)
            early_stop += 1

            # Save model + early stop
            # Early_stop_operator is min or max
            if early_stop_metric_value <= best_early_stop_value:
                early_stop = 0
                best_early_stop_value = early_stop_metric_value
                torch.save({
                    "net": net.state_dict(),
                    "ood_metrics": ood_metric_dicts,
                    "accuracy": None,
                    "best_early_stop_value": best_early_stop_value,
                    "args": args,
                }, f'{checkpoints_folder}/model_{best_early_stop_value}.pth'
                )

                print('Early stop metric ' + str(args.early_stop_metric) + ' beaten. Now ' + str(best_early_stop_value))

            if args.use_scheduler:
                scheduler.step(early_stop_metric_value)

        if early_stop == args.early_stop:
            print("early_stop reached")
            break

    print('Done')
    return


if __name__ == "__main__":
    main()
