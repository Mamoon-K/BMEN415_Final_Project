from datasetup import load_data

train, val, test = load_data()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

features = ['HR', 'O2Sat', 'Temp', 'Resp', 'Age'] #what should be the baseline features? Predicting SepsisLabel
