import argparse
import numpy as np
import os
import time
from metrics import calc_metrics
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OODDataset
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from network import ResNet
from losses import get_loss, get_confidence


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--idd_name", default="retina")
    parser.add_argument("--mode", type=str, default='devries',
                        choices=['baseline', 'devries', 'oe'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['skeletal-age', 'mura', 'mimic-crx'])
    parser.add_argument("--network", type=str, default="resnet")
    # Hyper params
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--use_budget", type=bool, default=False)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--use_hint", type=bool, default=False)
    parser.add_argument("--hint_rate", type=float, default=None)
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)


    # Training params
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument('--early_stop_metric', type=str, default="fpr_at_95_tpr")
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--eval_start', type=int, default=1)

    # Misc
    parser.add_argument("--load_memory", type=bool, default=False, help="Load images into CPU")

    args = parser.parse_args()
    args = utils.compute_args(args)

    # Create dataloader according to experiments
    loader_args = {'name': args.idd_name,
                   'mode': 'idd',
                   'root_dir': args.root[args.idd_name],
                   'csv_file': 'btrain.csv',
                   'load_memory': args.load_memory}

    train_loader = DataLoader(OODDataset(**loader_args),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_true_loader = DataLoader(OODDataset(**dict(loader_args, **{'csv_file': 'btest.csv'})),
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4)

    test_false_loaders = {}
    for ood_name in args.ood_name:
        test_false_loaders[ood_name] = DataLoader(OODDataset(**{'name': ood_name,
                                                                   'mode': 'ood',
                                                                   'root_dir': args.root[ood_name],
                                                                   'csv_file': 'btest.csv',
                                                                   'load_memory': False
                                                                   }),
                                                  batch_size=16,
                                                  shuffle=False,
                                                  num_workers=4)

    net = ResNet(args).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.8)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    print("Network", args.network, 'mode', args.mode,
          "\nUse budget", args.use_budget, "Use hint", args.use_hint,
          "\nLambda", args.lmbda, "beta", args.beta, "hint_rate", args.hint_rate,
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
        for train_iter, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            images, labels = sample
            logits, confidence = net(images.cuda())
            total_loss, task_loss, confidence_loss = get_loss(logits, confidence, labels.cuda(), args)
            total_loss.backward()
            optimizer.step()

            print(
                "\r[Epoch {}][Step {}/{}] Loss: {:.2f} [Task: {:.2f}, Confidence: {:.2f}, lambda: {:.2f}], Lr: {:.2e}, ES: {}, {:.2f} m remaining".format(
                    epoch + 1,
                    train_iter,
                    int(len(train_loader.dataset) / args.batch_size),
                    total_loss.cpu().data.numpy(),
                    task_loss.cpu().data.numpy(),
                    confidence_loss.cpu().data.numpy(),
                    args.lmbda,
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
                predictions = []

                for test_iter, sample in enumerate(data_loader, 0):
                    images, labels = sample
                    logits, confidence = net(images.cuda())
                    pred = torch.argmax(logits, dim=-1).data.cpu().numpy()
                    confidence = get_confidence(logits, confidence, args)

                    predictions.append(pred == labels.data.cpu().numpy())
                    confidences.append(confidence)

                predictions = np.concatenate(predictions)
                confidences = np.concatenate(confidences)
                return predictions, confidences

            # In domain evaluation
            ind_predictions, ind_confidences = evaluate(test_true_loader)
            ind_labels = np.ones(ind_confidences.shape[0])

            accuracy = round(float(np.mean(ind_predictions)), 4)
            print(args.idd_name, accuracy, '% accuracy')

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
                    "accuracy": accuracy,
                    "best_early_stop_value": best_early_stop_value,
                    "args": args,
                }, f'{checkpoints_folder}/model_{best_early_stop_value}.pth'
                )

                print('Early stop metric ' + str(args.early_stop_metric) + ' beaten. Now ' + str(best_early_stop_value))

            if args.use_scheduler:
                scheduler.step(accuracy)

        if early_stop == args.early_stop:
            print("early_stop reached")
            break

    print('Done')
    return


if __name__ == "__main__":
    main()
