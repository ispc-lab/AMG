import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.nn import Module, Linear
from torch.nn import functional as F
from torch_geometric.data import Batch
from copy import deepcopy
from .encoders import get_encoder, GNN_graphpred, MLP
from .common import *
from utils import dihedral_utils, chemutils
import torch
import numpy as np
from torch_geometric.data import Batch
from rdkit import Chem
from copy import deepcopy
from torch_scatter import scatter_add
from utils.chemutils import self_square_dist, eig_coord_from_dist, kabsch_torch, rand_rotate
from utils.gen_utils import find_reference, get_feat, SetAtomNum
from utils.docking import scale_tensor_vina_score
from rdkit import Chem

    
    
class AMG(Module):

    def __init__(self, config, pocket_atom_feature_dim, ligand_atom_feature_dim, vocab, ckpt_ligand_path=None, ckpt_pocket_path=None, device='cpu'):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = device

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
        self.pocket_atom_emb = Linear(pocket_atom_feature_dim, config.hidden_channels)
        self.pocket_encoder = get_encoder(config.encoder)
        self.focal_mlp_pocket = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.dist_mlp = MLP(in_dim=pocket_atom_feature_dim + ligand_atom_feature_dim, out_dim=1, num_layers=2)
    
        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction='mean')
    
        # Interaction_Net
        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(config.dim_gnn * 4, config.dim_gnn),
            nn.ReLU(),
            nn.Linear(config.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(config.dim_gnn * 4, config.dim_gnn),
            nn.ReLU(),
            nn.Linear(config.dim_gnn, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(config.dim_gnn * 4, config.dim_gnn),
            nn.ReLU(),
            nn.Linear(config.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.energies_emb = Linear(3, config.hidden_channels)
        self.energies_recover = nn.Sequential(
            nn.Linear(config.hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))
        self.loss_fn = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()

        # load pretraining Encoder and Decoder of ligand
        if ckpt_ligand_path:
            ckpt_ligand = torch.load(ckpt_ligand_path, map_location=device) 
            self.ligand_atom_emb.load_state_dict(ckpt_ligand['ligand_atom_emb'])
            self.embedding.load_state_dict(ckpt_ligand['embedding'])   
            self.W.load_state_dict(ckpt_ligand['W'])  
            self.W_o.load_state_dict(ckpt_ligand['W_o'])
            self.ligand_encoder.load_state_dict(ckpt_ligand['ligand_encoder'])
            self.comb_head.load_state_dict(ckpt_ligand['comb_head'])
            self.alpha_mlp.load_state_dict(ckpt_ligand['alpha_mlp'])
            self.focal_mlp_ligand.load_state_dict(ckpt_ligand['focal_mlp_ligand'])

        # load pretraining Encoder of pocket
        if ckpt_pocket_path:
            ckpt_pocket = torch.load(ckpt_pocket_path, map_location=device)
            self.pocket_atom_emb.load_state_dict(ckpt_pocket['pocket_atom_emb'])   
            self.pocket_encoder.load_state_dict(ckpt_pocket['pocket_encoder'])

    def forward(self, pocket_pos, pocket_atom_feature, ligand_pos, ligand_atom_feature, batch_pocket, batch_ligand, batch, action):
        # Embedding pocket and ligand
        h_pocket = self.pocket_atom_emb(pocket_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)
     
        # Encode pocket and ligand
        h_ctx_ligand = self.ligand_encoder(node_attr=h_ligand, pos=ligand_pos, batch=batch_ligand)     
        h_ctx_pocket = self.pocket_encoder(node_attr=h_pocket, pos=pocket_pos, batch=batch_pocket)        

        # Encode interaction
        # print("action: " + str(action))
        # energies = torch.tensor([float(action), float(action), float(action)]).float().to(h_ctx_pocket.device)
        energies = torch.tensor(action).float().to(h_ctx_pocket.device)
        # print("energies: " + str(energies))
        energies_feat = self.energies_emb(-energies)
  
        # Compose ligand, pocket, and interaction feature
        h_ctx, h_ctx_pocket, h_ctx_ligand = compose_ligand_pocket_interaction(h_pocket=h_ctx_pocket,      
                                                  h_ligand=h_ctx_ligand,
                                                  batch_pocket=batch_pocket,
                                                  batch_ligand=batch_ligand,
                                                  energies_feat=energies_feat)
        
        # Predict the focal of ligand and pocket
        focal_pocket = self.focal_mlp_pocket(h_ctx_pocket)
        focal_ligand = self.focal_mlp_ligand(h_ctx_ligand)
        focal_pred = torch.cat([focal_pocket, focal_ligand], dim=0)
   
        return focal_ligand, focal_pocket, focal_pred, h_ctx_ligand, h_ctx_pocket, h_ctx

    def forward_motif(self, h_ctx_focal, current_wid, current_atoms_batch, n_samples=1):
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=current_atoms_batch)
        motif_hiddens = self.embedding(current_wid)
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_scores = F.softmax(pred_scores, dim=-1)
        _, preds = torch.max(pred_scores, dim=1)
        # random select n_samples in topk
        k = 5*n_samples
        select_pool = torch.topk(pred_scores, k, dim=1)[1]
        index = torch.randint(k, (select_pool.shape[0], n_samples))
        preds = torch.cat([select_pool[i][index[i]] for i in range(len(index))])

        idx_parent = torch.repeat_interleave(torch.arange(pred_scores.shape[0]), n_samples, dim=0).to(pred_scores.device)
        prob = pred_scores[idx_parent, preds]
        return preds, prob

    def forward_attach(self, mol_list, next_motif_smiles, device):
        cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail = chemutils.assemble(mol_list, next_motif_smiles)
        
        graph_data = Batch.from_data_list([chemutils.mol_to_graph_data_obj_simple(mol) for mol in cand_mols]).to(device)
        comb_pred = self.comb_head(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch).reshape(-1)
        slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(cand_batch.bincount(), dim=0)], dim=0)
        select = [(torch.argmax(comb_pred[slice_idx[i]:slice_idx[i + 1]]) + slice_idx[i]).item() for i in
                  range(len(slice_idx) - 1)]
        select_mols = [cand_mols[i] for i in select]
        new_atoms = [new_atoms[i] for i in select]
        one_atom_attach = [one_atom_attach[i] for i in select]
        intersection = [intersection[i] for i in select]
        return select_mols, new_atoms, one_atom_attach, intersection, attach_fail

    def forward_alpha(self, ligand_pos, ligand_atom_feature, batch_ligand, xy_index, rotatable):
        # Encode ligand
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)      
        h_ctx_ligand = self.ligand_encoder(node_attr=h_ligand, pos=ligand_pos, batch=batch_ligand) 
    
        hx, hy = h_ctx_ligand[xy_index[:, 0]], h_ctx_ligand[xy_index[:, 1]]
        h_mol = scatter_add(h_ctx_ligand, dim=0, index=batch_ligand)
        h_mol = h_mol[rotatable]
        if self.config.random_alpha:
            rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
            rand_alpha = rand_dist.sample(hx.shape).to(hx.device)
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol, rand_alpha], dim=-1))
        else:
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
        return alpha

    def get_loss(self, batch):
        pocket_pos=batch['pocket_pos']
        pocket_atom_feature=batch['pocket_atom_feature'].float()
        ligand_pos=batch['ligand_context_pos']
        ligand_atom_feature=batch['ligand_context_feature_full'].float()
        ligand_pos_torsion=batch['ligand_pos_torsion']
        ligand_atom_feature_torsion=batch['ligand_feature_torsion'].float()
        batch_pocket=batch['pocket_element_batch']
        batch_ligand=batch['ligand_context_element_batch']
        batch_ligand_torsion=batch['ligand_element_torsion_batch']

        loss_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Encode for generation
        h_pocket = self.pocket_atom_emb(pocket_atom_feature)   
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)  
        h_ctx_ligand = self.ligand_encoder(node_attr=h_ligand, pos=ligand_pos, batch=batch_ligand)    
        h_ctx_pocket = self.pocket_encoder(node_attr=h_pocket, pos=pocket_pos, batch=batch_pocket)    

        # Encode for torsion prediction
        if len(batch['y_pos']) > 0:
            h_ligand_torsion = self.ligand_atom_emb(ligand_atom_feature_torsion)
            h_ctx_ligand_torsion = self.ligand_encoder(node_attr=h_ligand_torsion, pos=ligand_pos_torsion, batch=batch_ligand_torsion) 
      
        # Encoder for interaction prediction
        h_ligand_full = self.ligand_atom_emb(batch['ligand_atom_feature_full'].float())  
        h_ligand_full = self.ligand_encoder(node_attr=h_ligand_full, pos=batch['ligand_pos'], batch=batch['ligand_element_batch'])    
      
        
        # Encode interaction energies
        energies = self.Interaction_encoder(h_ligand_full, h_ctx_pocket, batch['ligand_element_batch'], batch_pocket, batch)  

        # energies = torch.tensor([-5, -5, -5]).to(self.device)
        energies_feat = self.energies_emb(energies.float())

        # Loss_0: Vina score prediction 
        energy_loss = self.loss_fn(energies.sum(-1), batch['vina_score']) * 0.1
        energy_loss = self.loss_fn(scale_tensor_vina_score(energies.sum(-1)), scale_tensor_vina_score(batch['vina_score'])) * 10
        loss_list[0] = energy_loss.item()

        pre_energies = self.energies_recover(energies_feat)
        recover_loss = self.loss_fn(pre_energies, energies.float())
        loss_list[6] = recover_loss.item()
   
        # Encode complex (h_ctx and energy)
        h_ctx, h_ctx_pocket, h_ctx_ligand = compose_ligand_pocket_interaction(h_pocket=h_ctx_pocket,    
                                                  h_ligand=h_ctx_ligand,
                                                  batch_pocket=batch_pocket,
                                                  batch_ligand=batch_ligand,
                                                  energies_feat=energies_feat)
     
        # Input feature into decoder
        pred_scores, comb_pred, focal_ligand_pred, focal_pocket_pred, pred_dist = self.decoder(h_ctx, h_ctx_ligand, h_ctx_pocket, batch)

        # Loss_1: next motif prediction 
        pred_loss = self.pred_loss(pred_scores, batch['next_wid'])
        loss_list[1] = pred_loss.item()

        # Loss_2: attachment prediction
        if len(batch['cand_labels']) > 0:
            comb_loss = self.comb_loss(comb_pred, batch['cand_labels'].view(comb_pred.shape).float())
            loss_list[2] = comb_loss.item()
        else:
            comb_loss = 0

        # Loss_3: focal prediction
        focal_loss = self.focal_loss(focal_ligand_pred.reshape(-1), batch['ligand_frontier'].float()) +\
                     self.focal_loss(focal_pocket_pred.reshape(-1), batch['pocket_contact'].float())
        loss_list[3] = focal_loss.item()
        
        # Loss_4: distance matrix prediction
        if len(batch['true_dm']) > 0:
            dm_loss = self.dist_loss(pred_dist, batch['true_dm']) / 10
            loss_list[4] = dm_loss.item()
        else:
            dm_loss = 0

        # Loss_5: torsion prediction
        if len(batch['y_pos']) > 0:
            Hx = dihedral_utils.rotation_matrix_v2(batch['y_pos'])
            xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)

            hx, hy = h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 0]], h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 1]]
            h_mol = scatter_add(h_ctx_ligand_torsion, dim=0, index=batch['ligand_element_torsion_batch'])
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
            loss_list[5] = torsion_loss.item()
        else:
            torsion_loss = 0

        loss = energy_loss + pred_loss + comb_loss + focal_loss + dm_loss + torsion_loss + recover_loss
        return loss, loss_list

    def decoder(self, h_ctx, h_ctx_ligand, h_ctx_pocket, batch):
        # Encode for motif prediction
        h_ctx_focal = h_ctx[batch['current_atoms']]
        # Next motif prediction
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        motif_hiddens = self.embedding(batch['current_wid'])
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
  
        # Attachment prediction
        if len(batch['cand_labels']) > 0:
            cand_mols = batch['cand_mols']
            comb_pred = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
        else:
            comb_pred = 0

        # Focal prediction
        focal_ligand_pred, focal_pocket_pred = self.focal_mlp_ligand(h_ctx_ligand), self.focal_mlp_pocket(h_ctx_pocket)

        # Distance matrix prediction
        if len(batch['true_dm']) > 0:
            inputs = torch.cat([batch['pocket_atom_feature'].float()[batch['dm_pocket_idx']], batch['ligand_context_feature_full'].float()[batch['dm_ligand_idx']]], dim=-1)
            pred_dist = self.dist_mlp(inputs)
        else:
            pred_dist = 0

        return pred_scores, comb_pred, focal_ligand_pred, focal_pocket_pred, pred_dist

    def generate_first(self, batch, vocab, sample_size, pos_list, feat_list, action=None):
        focal_ligand, focal_pocket, focal_pred, h_ctx_ligand, h_ctx_pocket, h_ctx = self.forward(pocket_pos=batch['pocket_pos'],
                                                        pocket_atom_feature=batch['pocket_atom_feature'].float(),
                                                        ligand_pos=batch['ligand_context_pos'],
                                                        ligand_atom_feature=batch['ligand_context_feature_full'].float(),
                                                        batch_pocket=batch['pocket_element_batch'],
                                                        batch_ligand=batch['ligand_context_element_batch'],
                                                        batch=batch,
                                                        action=action)
           
        pocket_atom_feature = batch['pocket_atom_feature'].float()
        focus_score = torch.sigmoid(focal_pocket)
        self.device = pocket_atom_feature.device
        slice_idx = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(batch['pocket_element_batch'].bincount(), dim=0)])
        focal_id = []
        for j in range(len(slice_idx) - 1):
            focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
            focal_id.append(torch.argmax(focus.reshape(-1).float()).item() + slice_idx[j].item())
        focal_id = torch.tensor(focal_id)

        h_ctx_focal = h_ctx_pocket[focal_id]
        current_wid = torch.tensor([vocab.size()] *sample_size).to(self.device)
        next_motif_wid, motif_prob = self.forward_motif(h_ctx_focal, current_wid, torch.arange(sample_size).to(self.device))  

        mol_list = [Chem.MolFromSmiles(vocab.get_smiles(id)) for id in next_motif_wid]
        for j in range(sample_size):
            Chem.AllChem.EmbedMolecule(mol_list[j])
            Chem.AllChem.UFFOptimizeMolecule(mol_list[j])
            ligand_pos, ligand_feat = torch.tensor(mol_list[j].GetConformer().GetPositions()), get_feat(mol_list[j])
            feat_list.append(ligand_feat)
            # Set the initial positions with distance matrix
            reference_pos, reference_idx = find_reference(batch['pocket_pos'][slice_idx[j]:slice_idx[j + 1]], focal_id[j] - slice_idx[j])

            p_idx, l_idx = torch.cartesian_prod(torch.arange(4), torch.arange(len(ligand_pos))).chunk(2, dim=-1)
            p_idx = p_idx.squeeze(-1)
            l_idx = l_idx.squeeze(-1)
    
            d_m = self.dist_mlp(torch.cat([pocket_atom_feature[reference_idx[p_idx]], ligand_feat[l_idx].to(self.device)], dim=-1)).reshape(4,len(ligand_pos))

            d_m = d_m ** 2
            p_d, l_d = self_square_dist(reference_pos), self_square_dist(ligand_pos)

            D = torch.cat([torch.cat([p_d, d_m], dim=1), torch.cat([d_m.permute(1, 0), l_d.to(self.device)], dim=1)])
            coordinate = eig_coord_from_dist(D.cpu())
            new_pos, _, _ = kabsch_torch(coordinate[:len(reference_pos)], reference_pos.cpu(),
                                                coordinate[len(reference_pos):])
            new_pos = new_pos.to(self.device)
            
            # new_pos += (batch['ligand_center'].to(self.device) * 0.8 + torch.mean(reference_pos, dim=0)*0.2) - torch.mean(new_pos, dim=0)
      
            # print('ref', batch['ligand_center'][j*3:j*3+3])
            # print(torch.mean(new_pos, dim=0))
            # new_pos += (batch['ligand_center'][j*3:j*3+3].to(self.device) - torch.mean(new_pos, dim=0)) 
            new_pos += (batch['ligand_center'][j*3:j*3+3].to(self.device) - torch.mean(new_pos, dim=0)) * .8
            # print('new', new_pos)
            pos_list.append(new_pos)

        atom_to_motif = [{} for _ in range(sample_size)]
        motif_to_atoms = [{} for _ in range(sample_size)]
        motif_wid = [{} for _ in range(sample_size)]
        for j in range(sample_size):
            for k in range(mol_list[j].GetNumAtoms()):
                atom_to_motif[j][k] = 0
        for j in range(sample_size):
            motif_to_atoms[j][0] = list(np.arange(mol_list[j].GetNumAtoms()))
            motif_wid[j][0] = next_motif_wid[j].item()
        return mol_list, atom_to_motif, motif_to_atoms, motif_wid, pos_list, feat_list

    def generate_second(self, batch, vocab, sample_size, pos_list, feat_list, motif_id, finished, i, mol_list, atom_to_motif, motif_to_atoms, motif_wid, action=None):
        # try:
        repeats = torch.tensor([len(pos) for pos in pos_list])
        ligand_batch = torch.repeat_interleave(torch.arange(sample_size), repeats)

        focal_ligand, focal_pocket, focal_pred, h_ctx_ligand, h_ctx_pocket, h_ctx = self.forward(pocket_pos=batch['pocket_pos'].float(),
                                                        pocket_atom_feature=batch['pocket_atom_feature'].float(),
                                                        ligand_pos=torch.cat(pos_list, dim=0).float(),
                                                        ligand_atom_feature=torch.cat(feat_list, dim=0).float().to(self.device),
                                                        batch_pocket=batch['pocket_element_batch'],
                                                        batch_ligand=ligand_batch.to(self.device),
                                                        batch=batch,
                                                        action=action)
        self.device = focal_ligand.device
        focus_score = torch.sigmoid(focal_ligand)
        can_focus = focus_score > 0.3
        slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])

        current_atoms_batch, current_atoms = [], []
        for j in range(len(slice_idx) - 1):
            focus = focus_score[slice_idx[j]:slice_idx[j + 1]]     
            if torch.sum(can_focus[slice_idx[j]:slice_idx[j + 1]]) > 0 and ~finished[j]:
                sample_focal_atom = torch.multinomial(focus.reshape(-1).float(), 1)
                focal_motif = atom_to_motif[j][sample_focal_atom.item()]
                motif_id[j] = focal_motif
            else:
                finished[j] = True

            current_atoms.extend((np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist())
            current_atoms_batch.extend([j] * len(motif_to_atoms[j][motif_id[j]]))
            mol_list[j] = SetAtomNum(mol_list[j], motif_to_atoms[j][motif_id[j]])
        
        # Second step: next motif prediction
        current_wid = [motif_wid[j][motif_id[j]] for j in range(len(mol_list))]
        next_motif_wid, motif_prob = self.forward_motif(h_ctx_ligand[torch.tensor(current_atoms)],
                                                    torch.tensor(current_wid).to(self.device),
                                                    torch.tensor(current_atoms_batch).to(self.device), n_samples=5)

        # assemble
        next_motif_smiles = [vocab.get_smiles(id) for id in next_motif_wid]
        new_mol_list, new_atoms, one_atom_attach, intersection, attach_fail = self.forward_attach(mol_list, next_motif_smiles, self.device)
        for j in range(len(mol_list)):
            if ~finished[j] and ~attach_fail[j]:
                # num_new_atoms
                mol_list[j] = new_mol_list[j]
        rotatable = torch.logical_and(torch.tensor(current_atoms_batch).bincount() == 2, torch.tensor(one_atom_attach))
        rotatable = torch.logical_and(rotatable, ~torch.tensor(attach_fail))
        rotatable = torch.logical_and(rotatable, ~finished)
        # update motif2atoms and atom2motif
        for j in range(len(mol_list)):
            if attach_fail[j] or finished[j]:
                continue
            motif_to_atoms[j][i] = new_atoms[j]
            motif_wid[j][i] = next_motif_wid[j]
            for k in new_atoms[j]:
                atom_to_motif[j][k] = i
      
        # generate initial positions
        for j in range(len(mol_list)):
            if attach_fail[j] or finished[j]:
                continue
            mol = mol_list[j]
            anchor = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
            # positions = mol.GetConformer().GetPositions()
            anchor_pos = deepcopy(pos_list[j][anchor])

            Chem.SanitizeMol(mol)
            Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
            Chem.AllChem.UFFOptimizeMolecule(mol)

            anchor_pos_new = mol.GetConformer().GetPositions()[anchor]
            new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
            '''
                    R, T = kabsch(np.matrix(anchor_pos), np.matrix(anchor_pos_new))
                    new_pos = R * np.matrix(mol.GetConformer().GetPositions()[new_idx]).T + np.tile(T, (1, len(new_idx)))
                    new_pos = np.array(new_pos.T)'''
            new_pos = mol.GetConformer().GetPositions()[new_idx]
            # new_pos, _, _ = kabsch_torch(torch.tensor(anchor_pos_new).to(h_ctx_focal.device), anchor_pos, torch.tensor(new_pos).to(h_ctx_focal.device))
            new_pos, _, _ = kabsch_torch(torch.tensor(anchor_pos_new), anchor_pos.cpu(), torch.tensor(new_pos))
            new_pos = new_pos.to(self.device)
            conf = mol.GetConformer()
            # update curated parameters
            pos_list[j] = torch.cat([pos_list[j], new_pos])
            feat_list[j] = get_feat(mol_list[j])
            
            for node in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(node, np.array(pos_list[j][node].cpu()))
            assert mol.GetNumAtoms() == len(pos_list[j])
            
        # predict alpha and rotate (only change the position)
        if torch.sum(rotatable) > 0 and i >= 2:
            repeats = torch.tensor([len(pos) for pos in pos_list])
            ligand_batch = torch.repeat_interleave(torch.arange(len(pos_list)), repeats)
            slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])
            xy_index = [(np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist() for j in range(len(slice_idx) - 1) if rotatable[j]]
            alpha = self.forward_alpha(ligand_pos=torch.cat(pos_list, dim=0).float().to(self.device),
                                                ligand_atom_feature=torch.cat(feat_list, dim=0).float().to(self.device),
                                                batch_ligand=ligand_batch.to(self.device), 
                                                xy_index=torch.tensor(xy_index).to(self.device),
                                                rotatable=rotatable)

            rotatable_id = [id for id in range(len(mol_list)) if rotatable[id]]
            xy_index = [motif_to_atoms[j][motif_id[j]] for j in range(len(slice_idx) - 1) if rotatable[j]]
            x_index = [intersection[j] for j in range(len(slice_idx) - 1) if rotatable[j]]
            y_index = [(set(xy_index[k]) - set(x_index[k])).pop() for k in range(len(x_index))]

            for j in range(len(alpha)):
                mol = mol_list[rotatable_id[j]]
                new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                positions = deepcopy(pos_list[rotatable_id[j]]).cpu()

                xn_pos = positions[new_idx].float().cpu()
                
                xn_pos = rand_rotate((positions[x_index[j]] - positions[y_index[j]]).reshape(-1), positions[x_index[j]].reshape(-1), xn_pos, alpha[j].cpu())
                if xn_pos.shape[0] > 0:
                    pos_list[rotatable_id[j]][-len(xn_pos):] = xn_pos
                conf = mol.GetConformer()
                for node in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(node, np.array(pos_list[rotatable_id[j]][node].cpu()))
                assert mol.GetNumAtoms() == len(pos_list[rotatable_id[j]])
   
        return mol_list, pos_list

    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
        :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(alpha.device)

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

    def Interaction_encoder(self, h_ligand, h_pocket, batch_ligand, batch_pocket, batch):
        inter_ligand_valid=batch['inter_ligand_valid']
        inter_target_valid=batch['inter_target_valid']
        ligand_vdw_radii=batch['inter_ligand_vdw_radii']
        target_vdw_radii=batch['inter_target_vdw_radii']
        rotor=batch['inter_rotor']
        hbond_interaction_indice=batch['inter_hbond_interaction_indice']
        hydrophobic_interaction_indice=batch['inter_hydrophobic_interaction_indice']

        h_ligand, h_pocket, inter_target_valid, target_vdw_radii, hbond_interaction_indice, hydrophobic_interaction_indice = compose_ligand_pocket(h_ligand, 
                                                                h_pocket, 
                                                                batch_ligand, 
                                                                batch_pocket, 
                                                                inter_target_valid, 
                                                                target_vdw_radii, 
                                                                hbond_interaction_indice, 
                                                                hydrophobic_interaction_indice)
        
        h1_ = h_ligand.unsqueeze(2).repeat(1, 1, h_pocket.size(1), 1)
        h2_ = h_pocket.unsqueeze(1).repeat(1, h_ligand.size(1), 1, 1)
        h_cat = torch.cat([h1_, h2_], -1)
     
        # compute energy component
        energies = []

        # Cal distance metric
        dm = self.cal_distance_matrix(batch['inter_ligand_pos'], batch['inter_pocket_pos'], self.config.dm_min)
        
        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            inter_ligand_valid,
            inter_target_valid,
        )
        energies.append(vdw_energy)
       
        # hbond interaction
        hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            hbond_interaction_indice,
        )
        energies.append(hbond)

        hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            hydrophobic_interaction_indice,
        )
        # print(hydrophobic)
        # print(hydrophobic_interaction_indice)
        energies.append(hydrophobic)
       
        energies = torch.cat(energies, -1)
  
        # rotor penalty
        if not self.config.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )   
        return energies
    
    def cal_hbond(self, dm, h, ligand_vdw_radii, target_vdw_radii, A):
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.config.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_hydrophobic(self, dm, h, ligand_vdw_radii, target_vdw_radii, A):
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.config.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
       
        return retval

    def cal_vdw_interaction(self, dm, h, ligand_vdw_radii, target_vdw_radii, ligand_valid, target_valid):
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(1, 1, target_valid.size(1))
        target_valid_ = target_valid.unsqueeze(1).repeat(1, ligand_valid.size(1), 1)
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.config.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.config.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.config.max_vdw_interaction - self.config.min_vdw_interaction)
        A = A + self.config.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        
        return energy

    def cal_distance_matrix(self, ligand_pos, target_pos, dm_min):
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)

        return dm
  

