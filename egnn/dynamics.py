import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn.egnn import EGNN, GNN
from egnn.egnn_sh import EGNN_Spherical
import numpy as np
import math
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean, scatter_add
from typing import Optional, Tuple
import warnings

class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim=256, scale=1.0):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)
        self.b = nn.Parameter(torch.rand(embed_dim), requires_grad=False)

    def forward(self, noise_level):
        x_proj = noise_level[:, None] * self.W[None, :] + self.b[None, :]
        return torch.cos(2 * torch.pi * x_proj)

class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_spherical',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, out_node_nf=None, lmax=2, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 num_rbf=16, rbf_type='bessel',
                 max_correlation_order=3,
                 weight_mode='True',
                 avg_num_neighbors=10.0, max_radius=10.0,
                 k_neighbors=16, k_neighbors_max=32,
                 scale_pocket_coords=True):
        super().__init__()
        self.mode = mode
        self.update_pocket_coords = update_pocket_coords
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim
        self.out_node_nf = out_node_nf
        self.lmax = lmax
        self.num_rbf = num_rbf
        self.weight_mode = weight_mode
        self.max_correlation_order = max_correlation_order
        self.rbf_type = rbf_type
        self.scale_pocket_coords = scale_pocket_coords
        self.k_neighbors = k_neighbors
        self.k_neighbors_max = k_neighbors_max

        self.time_emb_x = FourierTimeEmbedding(hidden_nf)
        self.time_emb_h = FourierTimeEmbedding(hidden_nf)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf),
            nn.LayerNorm(hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
        )

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )

        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        if condition_time:
            dynamics_node_nf = joint_nf + hidden_nf
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant
            )
            self.node_nf = dynamics_node_nf

        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
                device=device, act_fn=act_fn, n_layers=n_layers,
                attention=attention, normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
        
        elif mode == 'egnn_spherical':
            total_types = atom_nf + residue_nf
            self.egnn_spherical = EGNN_Spherical(
                in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers,
                lmax=lmax, num_rbf=self.num_rbf,
                rbf=self.rbf_type, trainable_rbf=True, 
                max_correlation_order=self.max_correlation_order,
                weight_mode=weight_mode,
                num_atom_types=total_types,
                avg_num_neighbors=avg_num_neighbors, max_radius=max_radius
            )

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, current_scale=None, lig_type_posterior=None):

        # Ligand        
        x_atoms_norm = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()   
        one_hot_atoms = xh_atoms[:, self.n_dims:]

        if lig_type_posterior is not None:
            # lig_type_posterior: [N_atoms, atom_nf], on probability simplex [0,1]
            ligand_attrs = lig_type_posterior
        else:
            # Fallback: uniform attrs (original behavior for backward compatibility)
            ligand_attrs = torch.ones_like(one_hot_atoms) * 1.0
        # Pocket
        x_residues_input = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()
        one_hot_residues = xh_residues[:, self.n_dims:]

        if t.dim() == 2 and t.shape[1] == 2:
            t_x = t[:, 0]
            t_h = t[:, 1]
        else:
            t_x = t.view(-1)    
            t_h = t.view(-1)

        emb_x = self.time_emb_x(t_x)
        emb_h = self.time_emb_h(t_h)

        t_concat = torch.cat([emb_x, emb_h], dim=-1)
        t_emb = self.time_mlp(t_concat)

        batch_idx_residues = mask_residues.long()
        t_emb_residues = t_emb[batch_idx_residues]

        batch_idx = mask_atoms.long() 
        t_emb_nodes = t_emb[batch_idx]

        h_atoms = self.atom_encoder(h_atoms)
        h_atoms = torch.cat([h_atoms, t_emb_nodes], dim=-1)
        h_residues = self.residue_encoder(h_residues)
        h_residues = torch.cat([h_residues, t_emb_residues], dim=-1)

        PAD_VAL = 0.0
        pocket_attrs = (one_hot_residues + 1.0) / 2.0

        pad_for_atoms = torch.full(
            (one_hot_atoms.shape[0], one_hot_residues.shape[1]), PAD_VAL,
            device=one_hot_atoms.device
        )
        pad_for_residues = torch.full(
            (one_hot_residues.shape[0], one_hot_atoms.shape[1]), PAD_VAL,
            device=one_hot_residues.device
        )

        atoms_attrs_unified = torch.cat([ligand_attrs, pad_for_atoms], dim=1)
        residues_attrs_unified = torch.cat([pad_for_residues, pocket_attrs], dim=1)
        
        node_attrs_input = torch.cat([atoms_attrs_unified, residues_attrs_unified], dim=0)

        if current_scale is not None:
            scale_atoms = current_scale[mask_atoms.long()] 
            if scale_atoms.dim() == 1:
                scale_atoms = scale_atoms.unsqueeze(1)     

            scale_residues = current_scale[mask_residues.long()]
            if scale_residues.dim() == 1:
                scale_residues = scale_residues.unsqueeze(1)
   
        else:
            scale_atoms = torch.ones_like(x_atoms_norm[:, :1])
            scale_residues = torch.ones_like(x_residues_input[:, :1])

        if self.scale_pocket_coords:
            x_residues_norm = x_residues_input
            x_residues_phys = x_residues_input * scale_residues
            rbf_scale = torch.cat([scale_atoms, scale_residues], dim=0)

        else:
            x_residues_phys = x_residues_input
            x_residues_norm = x_residues_input / scale_residues
            scale_pocket = torch.ones((x_residues_phys.shape[0], 1), device=self.device)
            rbf_scale = torch.cat([scale_atoms, scale_pocket], dim=0)

        #Scale-Ligand
        x_atoms_phys = x_atoms_norm * scale_atoms
        #x_atoms_phys_for_edges = x_atoms_phys * scale_atoms

        x = torch.cat((x_atoms_norm, x_residues_norm), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])
        
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms_phys, x_residues_phys, noise_scale=current_scale)
        assert torch.all(mask[edges[0]] == mask[edges[1]])
              
        # Get edge types
        if self.edge_nf > 0:
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        if self.mode == 'egnn_dynamics':
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)
            h_final, x_final = self.egnn(h, x, edges,
                                         update_coords_mask=update_coords_mask,
                                         batch_mask=mask, edge_attr=edge_types)
            vel = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]
        
        elif self.mode == 'egnn_spherical':
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)
            h_final, x_final = self.egnn_spherical(
                h, x, edges, t=None, node_attrs=node_attrs_input,
                update_coords_mask=update_coords_mask,
                batch_mask=mask, edge_attr=edge_types, time_emb=t_emb, rbf_scale=rbf_scale
            )
            vel = x_final - x
        
        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            feature_dim = self.atom_encoder[-1].out_features
            h_final = h_final[:, :feature_dim]

        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final = torch.nan_to_num(h_final, nan=0.0)
            else:
                print("Warning: NaN detected during validation. Ignoring this sample.")
                vel = torch.nan_to_num(vel, nan=0.0)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)
        
    def get_edges(
        self,
        batch_mask_ligand: torch.Tensor,
        batch_mask_pocket: torch.Tensor,
        x_ligand: torch.Tensor,
        x_pocket: torch.Tensor,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if torch.isnan(x_ligand).any() or torch.isinf(x_ligand).any():
            raise ValueError("FATAL: x_ligand contains NaN or Inf BEFORE get_edges!")
        if torch.isnan(x_pocket).any() or torch.isinf(x_pocket).any():
            raise ValueError("FATAL: x_pocket contains NaN or Inf BEFORE get_edges!")
            
        if batch_mask_ligand.max() != batch_mask_pocket.max():
             pass 

        device = x_ligand.device
        n_lig = x_ligand.shape[0]
        n_poc = x_pocket.shape[0]
        
        # LIGAND-LIGAND 
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_ligand.fill_diagonal_(False)
        
        if n_lig > 100 and self.edge_cutoff_l is not None:
             dist_lig = torch.cdist(x_ligand, x_ligand)
             adj_ligand = adj_ligand & (dist_lig <= max(self.edge_cutoff_l, 15.0))
        
        row_lig, col_lig = torch.where(adj_ligand)

        # --- 3. POCKET-POCKET (Sparse) ---
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_pocket.fill_diagonal_(False)
        
        if self.edge_cutoff_p is not None:
            dist_pocket = torch.cdist(x_pocket, x_pocket)
            # Thêm eps để an toàn tuyệt đối
            dist_pocket = torch.nan_to_num(dist_pocket, nan=9999.0) 
            adj_pocket = adj_pocket & (dist_pocket > 1e-5) 
            adj_pocket = adj_pocket & (dist_pocket <= self.edge_cutoff_p)
            
        row_poc, col_poc = torch.where(adj_pocket)
        row_poc += n_lig
        col_poc += n_lig

        # --- 4. CROSS EDGES (KNN) ---
        # Gọi hàm helper đã được gia cố
        lig_idx, poc_idx = self._build_knn_cross_indices(
            x_ligand, x_pocket,
            batch_mask_ligand, batch_mask_pocket,
            noise_scale
        )
        
        # Shift index
        poc_idx_shifted = poc_idx + n_lig
        
        # Combine
        edge_src = torch.cat([row_lig, row_poc, lig_idx, poc_idx_shifted])
        edge_dst = torch.cat([col_lig, col_poc, poc_idx_shifted, lig_idx])
        
        edges = torch.stack([edge_src, edge_dst], dim=0)
        return edges

    def _build_knn_cross_indices(
        self,
        x_ligand: torch.Tensor,
        x_pocket: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_pocket: torch.Tensor,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = x_ligand.device
        
        all_lig_indices = []
        all_poc_indices = []
        
        base_k = self.k_neighbors
        if noise_scale is not None:
            max_noise = noise_scale.max().item() if noise_scale.numel() > 0 else 0
            adaptive_k = base_k + int(max_noise / 10)
            k_target = min(adaptive_k, self.k_neighbors_max)
        else:
            k_target = base_k

        unique_batches = torch.unique(batch_ligand)
        
        for b_idx in unique_batches:
            mask_l = (batch_ligand == b_idx)
            mask_p = (batch_pocket == b_idx)
            
            if not mask_p.any():
                continue

            global_idx_l = torch.where(mask_l)[0]
            global_idx_p = torch.where(mask_p)[0]
            
            x_l_batch = x_ligand[mask_l] # [N_lig_local, 3]
            x_p_batch = x_pocket[mask_p] # [N_poc_local, 3]

            dist_mat = torch.cdist(x_l_batch, x_p_batch)
            
            current_k = min(k_target, x_p_batch.shape[0])
            
            if current_k <= 0: continue

            dist_vals, local_p_indices = torch.topk(dist_mat, k=current_k, dim=1, largest=False)
            
            # 5. Distance Cutoff (Optional)
            if self.edge_cutoff_i is not None:
                if noise_scale is not None:
                    eff_cutoff = self.edge_cutoff_i * (1 + max_noise / 20)
                    eff_cutoff = min(eff_cutoff, 30.0)
                else:
                    eff_cutoff = self.edge_cutoff_i * 1.5
                
                mask_dist = dist_vals <= eff_cutoff
            else:
                mask_dist = torch.ones_like(dist_vals, dtype=torch.bool)
                
            row_idx, col_idx = torch.where(mask_dist)
            
            final_lig_idx = global_idx_l[row_idx]

            target_local_p_idx = local_p_indices[row_idx, col_idx]
            final_poc_idx = global_idx_p[target_local_p_idx]
            
            all_lig_indices.append(final_lig_idx)
            all_poc_indices.append(final_poc_idx)

        
        if len(all_lig_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
            
        ligand_idx_total = torch.cat(all_lig_indices)
        pocket_idx_total = torch.cat(all_poc_indices)
        
        return ligand_idx_total, pocket_idx_total