import numpy as np
import torch
import operator
import os
import glob
import re

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


