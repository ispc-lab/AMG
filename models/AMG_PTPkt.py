import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from copy import deepcopy

from .encoders import get_encoder, GNN_graphpred, MLP
from .common import *
from utils import dihedral_utils, chemutils

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
    
def perturb(positions, mu=0, sigma=0.3):
    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(device)

    return positions_perturb


class AMG_PTPkt(Module):

    def __init__(self, config, pocket_atom_feature_dim):
        super().__init__()
        self.config = config
        self.pocket_atom_emb = Linear(pocket_atom_feature_dim, config.hidden_channels)
        self.pocket_encoder = get_encoder(config.encoder)        
        self.criterion = nn.BCEWithLogitsLoss()

    def get_loss(self, pocket_pos, pocket_atom_feature, batch_pocket, batch):
        self.device = pocket_pos.device
        h_pocket = self.pocket_atom_emb(pocket_atom_feature)

        # Encode for atom noise prediction
        h = self.pocket_encoder(node_attr=h_pocket, pos=pocket_pos, batch=batch_pocket)  # (N_p+N_l, H)

        pocket_pos_noise = perturb(pocket_pos)
        h_noise = self.pocket_encoder(node_attr=h_pocket, pos=pocket_pos_noise, batch=batch_pocket)  # (N_p+N_l, H)

        # if normalize:
        h = F.normalize(h, dim=-1)
        h_noise = F.normalize(h_noise, dim=-1)

        B = len(h)

        molecule_3D_repr_01_pos = h
        molecule_3D_repr_02_pos = h_noise
        molecule_3D_repr_01_neg = molecule_3D_repr_01_pos.repeat((1, 1))
        molecule_3D_repr_02_neg = torch.cat([h_noise[cycle_index(B, i + 1)] for i in range(1)], dim=0)

        pred_pos = torch.sum(molecule_3D_repr_01_pos * molecule_3D_repr_02_pos, dim=1)
        pred_neg = torch.sum(molecule_3D_repr_01_neg * molecule_3D_repr_02_neg, dim=1)

        loss_pos = self.criterion(pred_pos.double(), torch.ones(B).to(self.device).double())
        loss_neg = self.criterion(pred_neg.double(), torch.zeros(B * 1).to(self.device).double())

        SSL_loss = (loss_pos + 1 * loss_neg) / (1 + 1)
     
        return SSL_loss

   