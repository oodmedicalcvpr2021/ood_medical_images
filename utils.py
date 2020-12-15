import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def compute_args(args):
    # Root
    args.root = {'skeletal-age': 'data/skeletal-age/',
                 'retina': 'data/retina/',
                 'mura': 'data/mura/',
                 'drimdb': 'data/drimdb/',
                 'mimic-crx': 'data/mimic-crx/'
                 }
    # Num classes
    if args.idd_name == 'mura':
        args.num_classes = 2
    elif args.idd_name == 'retina':
        args.num_classes = 5
    elif args.idd_name == 'mimic-crx':
        args.num_classes = 2
    else:
        raise NotImplementedError

    return args


def plot_devries_histograms(ind_scores, ood_scores, name='histogram'):
    # Plot histogram of correctly classified and misclassified examples in visdom
    scores = np.concatenate([ind_scores, ood_scores])
    ranges = (np.min(scores), np.max(scores))
    plt.figure()
    sns.histplot(ind_scores.ravel(), binrange=ranges, kde=False, bins=50, stat="density",
                 label='In-distribution', color="#AECFBA")

    sns.histplot(ood_scores.ravel(), binrange=ranges, kde=False, bins=50, stat="density",
                 label='Out-of-distribution', color="#AABAD7")

    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title(name)
    plt.legend()

    return plt
