import torch
import torch.nn as nn
from torch.nn import Module, Linear
from torch.nn import functional as F
from torch_scatter import scatter_add
from .encoders import get_encoder, GNN_graphpred, MLP
from .common import *
from utils import dihedral_utils

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
    
def perturb(positions, mu=0, sigma=0.3):
    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(device)

    return positions_perturb

    
class AMG_PTLig(Module):
    def __init__(self, config, ligand_atom_feature_dim, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.W = nn.Linear(2 * config.hidden_channels, config.hidden_channels)
        self.W_o = nn.Linear(config.hidden_channels, self.vocab.size())
        self.ligand_encoder = get_encoder(config.encoder)
        self.comb_head = GNN_graphpred(num_layer=3, emb_dim=config.hidden_channels, num_tasks=1, JK='last',
                                       drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')
        if config.random_alpha:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 4, out_dim=1, num_layers=2)
        else:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=1, num_layers=2)
        self.focal_mlp_ligand = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.dist_mlp = MLP(in_dim=ligand_atom_feature_dim, out_dim=1, num_layers=2)
      
        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction='mean')

        self.criterion = nn.BCEWithLogitsLoss()


    def get_loss(self, ligand_pos, ligand_atom_feature, batch_ligand, ligand_pos_torsion, ligand_atom_feature_torsion, batch_ligand_torsion, batch, num_neg=1):
        loss_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.device = ligand_pos.device

        ### Encoder for no noise 
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)     
        h = self.ligand_encoder(node_attr=h_ligand, pos=ligand_pos, batch=batch_ligand)  # (N_p+N_l, H)
        # Encode for atom noise prediction
        ligand_pos_noise = perturb(ligand_pos)
        h_noise = self.ligand_encoder(node_attr=h_ligand, pos=ligand_pos_noise, batch=batch_ligand)  # (N_p+N_l, H)
       
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
        SSL_loss = (loss_pos + 1 * loss_neg) / (1 + num_neg)
        
        loss_list[0] = SSL_loss.item()
        
      
        # Encode for motif prediction
        h_focal = h[batch['current_atoms']]
        node_hiddens = scatter_add(h_focal, dim=0, index=batch['current_atoms_batch'])
        motif_hiddens = self.embedding(batch['current_wid']) 
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_loss = self.pred_loss(pred_scores, batch['next_wid'])
        loss_list[1] = pred_loss.item()
            
        # Encode for torsion prediction
        if len(batch['y_pos']) > 0:
            h_ligand_torsion = self.ligand_atom_emb(ligand_atom_feature_torsion)
            h_ligand_torsion = self.ligand_encoder(node_attr=h_ligand_torsion, pos=ligand_pos_torsion, batch=batch_ligand_torsion)  # (N_p+N_l, H)
                
        # torsion prediction
        if len(batch['y_pos']) > 0:
            Hx = dihedral_utils.rotation_matrix_v2(batch['y_pos'])
            xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)

            hx, hy = h_ligand_torsion[batch['ligand_torsion_xy_index'][:, 0]], h_ligand_torsion[batch['ligand_torsion_xy_index'][:, 1]]
            h_mol = scatter_add(h_ligand_torsion, dim=0, index=batch['ligand_element_torsion_batch'])
            if self.config.random_alpha:
                rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
                rand_alpha = rand_dist.sample(hx.shape).to(self.device)
                alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol, rand_alpha], dim=-1))
            else:
                alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
            # rotate xn
            R_alpha = self.build_alpha_rotation(torch.sin(alpha).squeeze(-1), torch.cos(alpha).squeeze(-1))
            xn_pos = torch.matmul(R_alpha, xn_pos.permute(0, 2, 1)).permute(0, 2, 1)

            p_idx, q_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            pred_sin, pred_cos = dihedral_utils.batch_dihedrals(xn_pos[:, p_idx],
                                                 torch.zeros_like(y_pos).unsqueeze(1).repeat(1, 9, 1),
                                                 y_pos.unsqueeze(1).repeat(1, 9, 1),
                                                 yn_pos[:, q_idx])
            dihedral_loss = torch.mean(dihedral_utils.von_Mises_loss(batch['true_cos'], pred_cos.reshape(-1), batch['true_sin'], pred_cos.reshape(-1))[batch['dihedral_mask']])
            torsion_loss = -dihedral_loss
            loss_list[4] = torsion_loss.item()
        else:
            torsion_loss = 0

        # attachment prediction
        if len(batch['cand_labels']) > 0:
            cand_mols = batch['cand_mols']
            comb_pred = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
            comb_loss = self.comb_loss(comb_pred, batch['cand_labels'].view(comb_pred.shape).float())
            loss_list[2] = comb_loss.item()
        else:
            comb_loss = 0
        
        # focal prediction
        focal_ligand_pred = self.focal_mlp_ligand(h_ligand)
        focal_loss = self.focal_loss(focal_ligand_pred.reshape(-1), batch['ligand_frontier'].float())
        loss_list[3] = focal_loss.item()
 
        # loss = (pred_loss + comb_loss + focal_loss + torsion_loss) * 0.5 + SSL_loss
        loss = pred_loss + comb_loss + focal_loss + torsion_loss + SSL_loss
        # print(loss)
        return loss, loss_list
    
    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
        :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(self.device)

        if torch.is_tensor(alpha_cos):
            H_alpha[:, 1, 1] = alpha_cos
            H_alpha[:, 1, 2] = -alpha
            H_alpha[:, 2, 1] = alpha
            H_alpha[:, 2, 2] = alpha_cos
        else:
            H_alpha[:, 1, 1] = torch.cos(alpha)
            H_alpha[:, 1, 2] = -torch.sin(alpha)
            H_alpha[:, 2, 1] = torch.sin(alpha)
            H_alpha[:, 2, 2] = torch.cos(alpha)

        return H_alpha

