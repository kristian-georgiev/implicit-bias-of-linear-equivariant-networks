import torch
import e2cnn
from e2cnn import nn as e2nn
import numpy as np
import pickle

def postprocess(batch):
    G = e2cnn.group.dihedral_group(4)  # D_8
    s = e2cnn.gspaces.FlipRot2dOnR2(4)
    c = e2cnn.nn.FieldType(s, [s.trivial_repr])

    x, y = batch
    # classify 1 vs 5, ignore other digits
    x = x[(y == 1) + (y == 5)]
    y = y[(y == 1) + (y == 5)]
    y[y == 5] = -1
    y = y.reshape(-1, 1)
    reduced_bs = len(y)
    x_g = e2nn.GeometricTensor(x, c)    
    # apply group elements and flatten
    x_g = torch.stack([x_g.transform(g).tensor for g in G.elements]).permute([1, 2, 3, 4, 0]).reshape(reduced_bs, -1)
    return x_g, y

def get_training_dataframes(experiment_name, force_train):
    if not force_train:
        try:
            with open(f'data/training/{experiment_name}.pickle', 'rb') as f:
                dfs = pickle.load(f)
        except:
            print('Will train before plotting.')
            dfs = {}  # if training has not been done yet
    else:
            print('Will train before plotting.')
            dfs = {}
    return dfs