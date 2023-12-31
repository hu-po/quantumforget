import argparse
import os
import pprint
import random
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from hyperopt import fmin, hp, tpe
from pennylane import numpy as qnp
from pennylane.templates import RandomLayers
from tensorboardX import SummaryWriter
from tensorflow import keras
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# one model class with multiple "backbones"

# split MNIST into 2 datasets based on label

# function to train X epochs on X dataset

# hyperopt to sweep over all combinations of models/datasets

# final plot