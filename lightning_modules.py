import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from egnn.dynamics import EGNNDynamics
from equivariant_diffusion.edm import Diffusion as EDM

from dataset import ProcessedLigandPocketDataset
import utils
from utils import AlphaFoldLRScheduler
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import BasicMolecularMetrics, CategoricalDistribution, \
    MoleculeProperties
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.docking import smina_score
import matplotlib.pyplot as plt
import csv

class LigandPocketEDM(pl.LightningModule):
    def __init__(
            self,
            outdir,
            dataset,
            datadir,
            batch_size,
            lr,
            egnn_params: Namespace,
            diffusion_params,
            num_workers,
            augment_noise,
            augment_rotation,
            clip_grad,
            eval_epochs,
            eval_params,
            visualize_sample_epoch,
            visualize_chain_epoch,
            use_bond_loss,
            bond_loss_weight,
            auxiliary_loss,
            loss_params,
            mode,
            node_histogram,
            pocket_representation='CA',
            virtual_nodes=False
    ):
        super(LigandPocketEDM, self).__init__()
        self.save_hyperparameters()

        self.mode = mode
        assert pocket_representation in {'CA', 'full-atom','backbone'}
        self.pocket_representation = pocket_representation

        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.eval_epochs = eval_epochs
        self.visualize_sample_epoch = visualize_sample_epoch
        self.visualize_chain_epoch = visualize_chain_epoch
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.clip_grad = clip_grad
        if clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        self.lig_type_encoder = self.dataset_info['atom_encoder']
        self.lig_type_decoder = self.dataset_info['atom_decoder']
        self.pocket_type_encoder = self.dataset_info['aa_encoder'] \
            if self.pocket_representation == 'CA' or self.pocket_representation == 'backbone'\
            else self.dataset_info['atom_encoder']
        self.pocket_type_decoder = self.dataset_info['aa_decoder'] \
            if self.pocket_representation == 'CA' or self.pocket_representation == 'backbone' \
            else self.dataset_info['atom_decoder']

        smiles_list = None if eval_params.smiles_file is None \
            else np.load(eval_params.smiles_file)
        self.ligand_metrics = BasicMolecularMetrics(self.dataset_info,
                                                    smiles_list)
        self.molecule_properties = MoleculeProperties()
        self.ligand_type_distribution = CategoricalDistribution(
            self.dataset_info['atom_hist'], self.lig_type_encoder)
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.pocket_type_encoder)
        else:
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.virtual_nodes = virtual_nodes
        self.data_transform = None
        self.max_num_nodes = len(node_histogram) - 1
        if virtual_nodes:
            # symbol = 'virtual'
            symbol = 'Ne'  # visualize as Neon atoms
            self.lig_type_encoder[symbol] = len(self.lig_type_encoder)
            self.virtual_atom = self.lig_type_encoder[symbol]
            self.lig_type_decoder.append(symbol)
            self.data_transform = utils.AppendVirtualNodes(
                self.max_num_nodes, self.lig_type_encoder, symbol)

            # Update dataset_info dictionary. This is necessary for using the
            # visualization functions.
            self.dataset_info['atom_encoder'] = self.lig_type_encoder
            self.dataset_info['atom_decoder'] = self.lig_type_decoder

        self.atom_nf = len(self.lig_type_decoder)
        self.aa_nf = len(self.pocket_type_decoder)
        self.x_dims = 3
        atom_type_prior = self._compute_atom_type_prior()


        net_dynamics = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.aa_nf,
            mode=egnn_params.mode,
            n_dims=self.x_dims,
            joint_nf=egnn_params.joint_nf,
            device=egnn_params.device if torch.cuda.is_available() else 'cpu',
            hidden_nf=egnn_params.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            edge_cutoff_ligand=egnn_params.__dict__.get('edge_cutoff_ligand'),
            edge_cutoff_pocket=egnn_params.__dict__.get('edge_cutoff_pocket'),
            edge_cutoff_interaction=egnn_params.__dict__.get('edge_cutoff_interaction'),
            update_pocket_coords=(self.mode == 'joint'),
            reflection_equivariant=egnn_params.reflection_equivariant,
            edge_embedding_dim=egnn_params.__dict__.get('edge_embedding_dim'),
            num_rbf=egnn_params.__dict__.get('num_rbf', 16),
            rbf_type=egnn_params.__dict__.get('rbf_type', 'expnormal'), # 'bessel', 'gaussian'
            max_correlation_order=egnn_params.__dict__.get('max_correlation_order', 3),
            lmax=egnn_params.__dict__.get('lmax', 2),
            avg_num_neighbors=egnn_params.__dict__.get('avg_num_neighbors', 28.0),
            max_radius=egnn_params.__dict__.get('max_radius', 10.0),
            scale_pocket_coords=getattr(diffusion_params, 'scale_pocket_coords', True)
        )
        self.model = EDM(
                        dynamics=net_dynamics,
                        atom_nf=self.atom_nf,
                        residue_nf=self.aa_nf,
                        n_dims=self.x_dims,
                        size_histogram=node_histogram,
                        # --- Position ---
                        sigma_min_pos=getattr(diffusion_params, 'sigma_min_pos', 0.0004),
                        sigma_max_pos=getattr(diffusion_params, 'sigma_max_pos', 80.0),
                        sigma_data_pos=getattr(diffusion_params, 'sigma_data_pos', 2.5),
                        P_mean_pos=getattr(diffusion_params, 'P_mean_pos', -1.2),
                        P_std_pos=getattr(diffusion_params, 'P_std_pos', 1.5),
        
                        # --- Features ---
                        sigma_min_feat=getattr(diffusion_params, 'sigma_min_feat', 0.002),
                        sigma_max_feat=getattr(diffusion_params, 'sigma_max_feat', 40.0),
                        sigma_data_feat=getattr(diffusion_params, 'sigma_data_feat', 0.5),
                        P_mean_feat=getattr(diffusion_params, 'P_mean_feat', -1.2),
                        P_std_feat=getattr(diffusion_params, 'P_std_feat', 1.5),

                        rho=getattr(diffusion_params, 'rho', 7),
                        dilated_schedule=getattr(diffusion_params, 'dilated_schedule', False),
                        tau_start=getattr(diffusion_params, 'tau_start', 0.6),
                        tau_end=getattr(diffusion_params, 'tau_end', 0.8),
                        num_sampling_steps=getattr(diffusion_params, 'num_sampling_steps', 40),
                        norm_values=diffusion_params.normalize_factors,
                        virtual_node_idx=self.lig_type_encoder[symbol] if virtual_nodes else None,
                        scale_pocket_coords=getattr(diffusion_params, 'scale_pocket_coords', True),
                        atom_type_prior=atom_type_prior,
                        loss_x_weight=getattr(diffusion_params, 'loss_x_weight', 10.0)
                    )        


        self.auxiliary_loss = auxiliary_loss
        self.lj_rm = self.dataset_info['lennard_jones_rm']
        if self.auxiliary_loss:
            self.clamp_lj = loss_params.clamp_lj
            sigma_min = getattr(diffusion_params, 'sigma_min_pos', 0.002)
            sigma_max = getattr(diffusion_params, 'sigma_max_pos', 80.0)
            self.auxiliary_weight_schedule = EDMWeightSchedule(
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            max_weight=loss_params.max_weight
                        )
        self.use_bond_loss = use_bond_loss

        # Lấy các tham số sigma giống như mô hình chính
        sigma_min = getattr(diffusion_params, 'sigma_min_pos', 0.002)
        sigma_max = getattr(diffusion_params, 'sigma_max_pos', 80.0)
        
        # Tạo schedule: Max weight chính là giá trị cấu hình (0.1), min weight sẽ là 0
        self.bond_loss_schedule = EDMWeightSchedule(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            max_weight=bond_loss_weight # Trọng số tối đa (khi sigma -> 0)
        )     
        self.bond_loss_weight = bond_loss_weight


    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                        amsgrad=True, weight_decay=1e-12)
                        
            scheduler = AlphaFoldLRScheduler(
                        optimizer,
                        base_lr=1e-6,              # Bắt đầu rất nhỏ
                        max_lr=self.lr,            # Tăng lên mức LR trong config (1e-4)
                        warmup_no_steps=2000,      # Warmup trong 2000 steps đầu
                        start_decay_after_n_steps=50000, # Bắt đầu giảm sau step 100k
                        decay_every_n_steps=50000,        # Giảm mỗi 50k steps
                        decay_factor=0.95                 # Giảm 5% mỗi lần
                    )
                    
                    # 4. Trả về cấu hình cho Lightning
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # QUAN TRỌNG: Cập nhật mỗi step (batch)
                    "frequency": 1,
                    "monitor": "train_loss", 
                    "name": "AlphaFold_LR"   #
                }
            }
            

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'train.npz'), transform=self.data_transform)
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.npz'), transform=self.data_transform)
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.npz'), transform=self.data_transform)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn,
                    pin_memory=True)



    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=self.val_dataset.collate_fn,
                    pin_memory=True)
        


    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)
    

    def get_ligand_and_pocket(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE),
        }
        if self.virtual_nodes:
            ligand['num_virtual_atoms'] = data['num_virtual_atoms'].to(
                self.device, INT_TYPE)

        pocket = {
            'x': data['pocket_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['pocket_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return ligand, pocket

    def forward(self, data):
        ligand, pocket = self.get_ligand_and_pocket(data)

        loss, info = self.model(ligand, pocket)

        if self.use_bond_loss and 'lig_bonds' in data:
            
            if 'xh_lig_hat' in info:
                x_pred = info['xh_lig_hat'][:, :self.x_dims]
                x_true = ligand['x']
                
                # Gọi hàm tính loss
                bond_loss_raw = self.compute_bond_loss(
                    x_pred, 
                    x_true, 
                    data['lig_bonds']
                )
                current_sigma = info.get('sigma_x', 0.002)
                dynamic_weight = self.bond_loss_schedule(current_sigma)
                weighted_bond_loss = dynamic_weight * bond_loss_raw
                
                loss = loss + weighted_bond_loss
                
                info['bond_loss'] = weighted_bond_loss.item()

        if self.auxiliary_loss and self.training:
            xh_lig_hat = info['xh_lig_hat']
            sigmas = info['sigma_x']

            x_lig_hat = xh_lig_hat[:, :self.x_dims]
            h_lig_hat = xh_lig_hat[:, self.x_dims:]

            weights = self.auxiliary_weight_schedule(sigmas)
            lj_loss_per_mol = self.lj_potential(x_lig_hat, h_lig_hat, ligand['mask'])
            weighted_lj = weights * lj_loss_per_mol
            weighted_lj_mean = weighted_lj.mean()
            loss = loss + weighted_lj_mean
            info['weighted_lj'] = weighted_lj_mean.item()
        

        return loss, info

    def lj_potential(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Compute pair-wise potentials
        r = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1).sqrt()

        # Get optimal radii
        lennard_jones_radii = torch.tensor(self.lj_rm, device=r.device)
        # unit conversion pm -> A
        lennard_jones_radii = lennard_jones_radii / 100.0
        # normalization
        lennard_jones_radii = lennard_jones_radii / self.model.norm_values[0]
        atom_type_idx = atom_one_hot.argmax(1)
        rm = lennard_jones_radii[atom_type_idx[edges[0]],
                                 atom_type_idx[edges[1]]]
        sigma = 2 ** (-1 / 6) * rm
        out = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)

        if self.clamp_lj is not None:
            out = torch.clamp(out, min=None, max=self.clamp_lj)

        # Compute potential per atom
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))

        # Sum potentials of all atoms
        return scatter_add(out, batch_mask, dim=0)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):

        if self.augment_noise > 0:
            raise NotImplementedError
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian(x.size(), x.device)
            x = x + eps * args.augment_noise

        if self.augment_rotation:
            raise NotImplementedError
            x = utils.random_rotation(x).detach()

        try:
            loss, info = self.forward(data)
        
        
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e
            
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if 'debug/loss_x_high_noise' in info:
            self.log('H', info['debug/loss_x_high_noise'], prog_bar=True, logger=True, on_step=True)
            self.log('M', info['debug/loss_x_mid_noise'], prog_bar=True, logger=True, on_step=True)
            self.log('L', info['debug/loss_x_low_noise'], prog_bar=True, logger=True, on_step=True)
        info['loss'] = loss
        info.pop('xh_lig_hat', None)
        self.log_metrics(info, 'train', batch_size=len(data['num_lig_atoms']))

        return info

    def _shared_eval(self, data, prefix, *args):
        nll, info = self.forward(data)
        loss = nll

        info['loss'] = loss

        info.pop('xh_lig_hat', None)

        self.log_metrics(info, prefix, batch_size=len(data['num_lig_atoms']),
                         sync_dist=True)

        return info

    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)

    def on_validation_epoch_end(self):

        # Perform validation on single GPU
        if not self.trainer.is_global_zero:
            return

        suffix = '' if self.mode == 'joint' else '_given_pocket'

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()

            sampling_results = getattr(self, 'sample_and_analyze' + suffix)(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()
            getattr(self, 'sample_and_save' + suffix)(
                self.eval_params.n_visualize_samples)
            
            self.analyze_entropy_and_save(n_samples=1)
            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.keep_frames)
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    @torch.no_grad()
    def sample_and_analyze(self, n_samples, dataset=None, batch_size=None):
        print(f'Analyzing sampled molecules at epoch {self.current_epoch}...')

        if dataset is None:
            dataset = self.val_dataset

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            indices = torch.randint(len(dataset), (n_samples_batch,))
            batch = dataset.collate_fn([dataset[j] for j in indices])
            
            ligand, pocket = self.get_ligand_and_pocket(batch)

            pocket_for_model = {
                'x': pocket['x'].clone(),
                'one_hot': pocket['one_hot'].clone(),
                'size': pocket['size'].clone(),
                'mask': pocket['mask'].clone()
            }

            if self.virtual_nodes:
                num_nodes_lig = self.max_num_nodes
            else:
                num_nodes_lig = self.model.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])
            x_gen, h_gen = self.model.sample_denovo(pocket_for_model, num_nodes_lig)
            lig_mask = utils.num_nodes_to_batch_mask(len(num_nodes_lig), num_nodes_lig, self.device)
            x = x_gen.detach().cpu()
            atom_type = h_gen.argmax(1).detach().cpu() # Chuyển One-Hot về Index
            lig_mask = lig_mask.cpu()

            xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

            if self.virtual_nodes:
                vnode_mask = (atom_type == self.virtual_atom)
                x = x[~vnode_mask, :]
                atom_type = atom_type[~vnode_mask]
                lig_mask = lig_mask[~vnode_mask]

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def analyze_sample(self, molecules, atom_types, aa_types, receptors=None):
        # Distribution of node types
        kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
            if self.ligand_type_distribution is not None else -1
        kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
            if self.pocket_type_distribution is not None else -1

        # Convert into rdmols
        rdmols = [build_molecule(*graph, self.dataset_info) for graph in molecules]

        # Other basic metrics
        (validity, connectivity, uniqueness, novelty), (_, connected_mols) = \
            self.ligand_metrics.evaluate_rdmols(rdmols)

        qed, sa, logp, lipinski, diversity = \
            self.molecule_properties.evaluate_mean(connected_mols)

        out = {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
            'Validity': validity,
            'Connectivity': connectivity,
            'Uniqueness': uniqueness,
            'Novelty': novelty,
            'QED': qed,
            'SA': sa,
            'LogP': logp,
            'Lipinski': lipinski,
            'Diversity': diversity
        }

        # Simple docking score
        if receptors is not None:
            # out['smina_score'] = np.mean(smina_score(rdmols, receptors))
            valid_receptors = [r for r in receptors if r is not None]
            if len(connected_mols) > 0 and len(valid_receptors) > 0:
                try:
                    scores = smina_score(connected_mols, receptors)
                    mean_score = np.mean(scores)
                    out['smina_score'] = mean_score
                    
                    # --- THÊM DÒNG NÀY ĐỂ HIỆN RA MÀN HÌNH ---
                    print(f"✅ Smina Docking Score ({len(connected_mols)} mols): {mean_score:.4f}")
                    # -----------------------------------------
                except Exception as e:
                    print(f"Docking failed: {e}")
                    out['smina_score'] = np.nan
            else:
                out['smina_score'] = np.nan
                print("⚠️ Skipping Docking: No connected molecules found.")

        return out

    def get_full_path(self, receptor_name):
        if str(receptor_name).endswith('.pdb'):
            return Path(self.datadir, 'val', receptor_name)
        try:
            pdb, suffix = receptor_name.split('.')
            receptor_name = f'{pdb.upper()}-{suffix}.pdb'
            return Path(self.datadir, 'val', receptor_name)
        except ValueError:
            # Fallback nếu tên không có dấu chấm
            return Path(self.datadir, 'val', f"{receptor_name}.pdb")

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing sampled molecules given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        receptors = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )

            ligand, pocket = self.get_ligand_and_pocket(batch)

            pocket_for_model = {
                'x': pocket['x'].clone(),
                'one_hot': pocket['one_hot'].clone(),
                'size': pocket['size'].clone(),
                'mask': pocket['mask'].clone()
            }
            
            if 'receptors' in batch:
                receptors.extend([self.get_full_path(x) for x in batch['receptors']])
            else:
                # Fallback: Không có receptor info, skip docking score
                receptors.extend([None] * n_samples_batch)
            

            if self.virtual_nodes:
                num_nodes_lig = self.max_num_nodes
            else:
                num_nodes_lig = self.model.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])
            x_gen, h_gen = self.model.sample_denovo(pocket_for_model, num_nodes_lig)
            lig_mask = utils.num_nodes_to_batch_mask(len(num_nodes_lig), num_nodes_lig, self.device)
            x = x_gen.detach().cpu()
            atom_type = h_gen.argmax(1).detach().cpu()
            lig_mask = lig_mask.cpu()
            xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
                
            if self.virtual_nodes:
                # Remove virtual nodes for analysis
                vnode_mask = (atom_type == self.virtual_atom)
                x = x[~vnode_mask, :]
                atom_type = atom_type[~vnode_mask]
                lig_mask = lig_mask[~vnode_mask]

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types,
                                   receptors=receptors)

    def sample_and_save(self, n_samples):

        indices = torch.randint(len(self.val_dataset), (n_samples,))
        batch = self.val_dataset.collate_fn([self.val_dataset[i] for i in indices])
        
        ligand, pocket = self.get_ligand_and_pocket(batch)
        pocket_for_model = {
            'x': pocket['x'].clone(),
            'one_hot': pocket['one_hot'].clone(),
            'size': pocket['size'].clone(),
            'mask': pocket['mask'].clone()
        }
        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.model.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])
        
        x_gen, h_gen = self.model.sample_denovo(pocket_for_model, num_nodes_lig)
        xh_lig = torch.cat([x_gen, h_gen], dim=1)
        xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        lig_mask = utils.num_nodes_to_batch_mask(len(num_nodes_lig), num_nodes_lig, self.device)
        pocket_mask = pocket['mask']

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.lig_type_encoder)
        elif self.pocket_representation == 'backbone':
            x_pocket = xh_pocket[:, :self.x_dims]
            c_idx = self.lig_type_encoder.get('C', 0) 
            one_hot_pocket = torch.zeros(
                (x_pocket.shape[0], len(self.lig_type_encoder)), 
                device=x_pocket.device
            )
            one_hot_pocket[:, c_idx] = 1.0
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.lig_type_decoder,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_and_save_given_pocket(self, n_samples):
        batch = self.val_dataset.collate_fn(
            [self.val_dataset[i] for i in torch.randint(len(self.val_dataset),
                                                        size=(n_samples,))]
        )
        ligand, pocket = self.get_ligand_and_pocket(batch)
        
        pocket_for_model = {
            'x': pocket['x'].clone(),
            'one_hot': pocket['one_hot'].clone(),
            'size': pocket['size'].clone(),
            'mask': pocket['mask'].clone()
        }

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.model.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])
        x_gen, h_gen = self.model.sample_denovo(pocket_for_model, num_nodes_lig)
        xh_lig = torch.cat([x_gen, h_gen], dim=1)
        xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        lig_mask = utils.num_nodes_to_batch_mask(len(num_nodes_lig), num_nodes_lig, self.device)
        pocket_mask = pocket['mask']

        if self.pocket_representation in ['CA', 'backbone']:
            # CA hoặc Backbone: Chuyển từ AA Type -> Atom Type (C, N, O...)
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], 
                self.lig_type_encoder,
                representation=self.pocket_representation
            )
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.lig_type_decoder,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_chain_and_save_given_pocket(self, keep_frames):
        n_samples = 1

        batch = self.val_dataset.collate_fn([
            self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]
        ])
        ligand, pocket = self.get_ligand_and_pocket(batch)

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.model.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        chain_x, chain_h = self.model.sample_denovo({
                'x': pocket['x'].clone(),
                'one_hot': pocket['one_hot'].clone(),
                'size': pocket['size'].clone(),
                'mask': pocket['mask'].clone(),
            }, num_nodes_lig, return_frames=keep_frames)
        
        chain_lig = torch.cat([chain_x, chain_h], dim=2)
        xh_pocket_single = torch.cat([pocket['x'], pocket['one_hot']], dim=1).cpu()
        chain_pocket = xh_pocket_single.unsqueeze(0).repeat(chain_x.size(0), 1, 1)

        # Repeat last frame to see final sample better.
        chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                              dim=0)
        chain_pocket = torch.cat(
            [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

        # Prepare entire chain.
        x_lig = chain_lig[:, :, :self.x_dims]
        one_hot_lig = chain_lig[:, :, self.x_dims:]
        one_hot_lig = F.one_hot(
            torch.argmax(one_hot_lig, dim=2),
            num_classes=len(self.lig_type_decoder))
        
        x_pocket = chain_pocket[:, :, :3]
        one_hot_pocket_raw = chain_pocket[:, :, 3:]

        if self.pocket_representation in ['CA', 'backbone']:
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, 
                self.lig_type_encoder, 
                representation=self.pocket_representation
            )  
        else:
            # Full atom: Cần argmax -> one_hot để khớp format visualizer
            idx = torch.argmax(one_hot_pocket_raw, dim=2)
            one_hot_pocket = F.one_hot(idx, num_classes=len(self.pocket_type_decoder))

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.lig_type_decoder,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)
    
    @torch.no_grad()
    def analyze_entropy_and_save(self, n_samples=1):
        """
        Run sampling loop with entropy monitoring at each step.
        Uses the same dual-schedule + _denoise_step API as sample_denovo.
        """
        # 1. Get a validation sample
        batch = self.val_dataset.collate_fn([
            self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]
        ])
        ligand, pocket = self.get_ligand_and_pocket(batch)

        # 2. Setup Monitor
        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        outdir.mkdir(parents=True, exist_ok=True)
        monitor = ConfidenceMonitor(
            outdir,
            sigma_max=self.model.sigma_max_feat,
            sigma_min=self.model.sigma_min_feat,
            rho=self.model.rho,
        )

        # 3. Prepare pocket & noise schedules (mirrors sample_denovo)
        _, pocket = self.model.normalize(pocket=pocket)
        xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            num_nodes_lig = self.model.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        lig_mask = utils.num_nodes_to_batch_mask(
            len(num_nodes_lig), num_nodes_lig, self.device)
        batch_size = len(num_nodes_lig)

        # Dual sigma schedules
        sigmas_x_seq, sigmas_h_seq = self.model.get_sigma_schedules(
            self.model.num_sampling_steps)

        # Initial noise with separate sigma_x / sigma_h
        z_xh, xh_pocket = self.model.sample_normal_zero_com(
            lig_mask=lig_mask,
            pocket_mask=pocket['mask'],
            xh0_pocket=xh_pocket,
            sigma_x=sigmas_x_seq[0],
            sigma_h=sigmas_h_seq[0],
        )

        x_pocket = xh_pocket[:, :self.model.n_dims]
        h_pocket = xh_pocket[:, self.model.n_dims:]

        z_x = z_xh[:, :self.model.n_dims]
        z_h = z_xh[:, self.model.n_dims:]

        # 4. Sampling loop (Heun) — uses _denoise_step helper
        for i in range(self.model.num_sampling_steps):
            sigma_x_cur  = sigmas_x_seq[i].item()
            sigma_h_cur  = sigmas_h_seq[i].item()
            sigma_x_next = sigmas_x_seq[i + 1].item()
            sigma_h_next = sigmas_h_seq[i + 1].item()

            # Euler step — also gives us denoised estimate for monitoring
            d_x, d_h, grad_x, grad_h = self.model._denoise_step(
                z_x, z_h, x_pocket, h_pocket,
                sigma_x_cur, sigma_h_cur,
                lig_mask, pocket, batch_size, self.device,
            )

            # Record entropy of denoised h at this noise level
            monitor.update(d_h, sigma_h_cur)

            dt_x = sigma_x_next - sigma_x_cur
            dt_h = sigma_h_next - sigma_h_cur

            next_x = z_x + grad_x * dt_x
            next_h = z_h + grad_h * dt_h

            # Heun correction
            if sigma_x_next > 0 and sigma_h_next > 0:
                _, _, grad_x_2, grad_h_2 = self.model._denoise_step(
                    next_x, next_h, x_pocket, h_pocket,
                    sigma_x_next, sigma_h_next,
                    lig_mask, pocket, batch_size, self.device,
                )
                next_x = z_x + 0.5 * (grad_x + grad_x_2) * dt_x
                next_h = z_h + 0.5 * (grad_h + grad_h_2) * dt_h

            z_x = next_x
            z_h = next_h

            # Re-center — update x_pocket to stay consistent
            z_x, x_pocket = self.model.remove_mean_batch(
                z_x, x_pocket, lig_mask, pocket['mask'])

        # 5. Save entropy plot & CSV
        monitor.save_data(self.current_epoch)

    def prepare_pocket(self, biopython_residues, repeats=1):

        if self.pocket_representation == 'CA':
            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in biopython_residues]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[three_to_one(res.get_resname())]
                 for res in biopython_residues], device=self.device)
            
        elif self.pocket_representation == 'backbone':
            pocket_coords_list = []
            pocket_types_list = []
            target_atoms = ['N', 'CA', 'C', 'O']
            
            for res in biopython_residues:
                # Chỉ lấy residue nếu đủ 4 nguyên tử
                if all(a in res for a in target_atoms):
                    res_name = three_to_one(res.get_resname())
                    # Skip unknown AA
                    if res_name not in self.pocket_type_encoder:
                        continue
                        
                    aa_idx = self.pocket_type_encoder[res_name]
                    
                    for atom_name in target_atoms:
                        pocket_coords_list.append(res[atom_name].get_coord())
                        pocket_types_list.append(aa_idx)
            
            pocket_coord = torch.tensor(np.array(pocket_coords_list), device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(pocket_types_list, device=self.device)
            
            pocket_one_hot = F.one_hot(
                pocket_types, num_classes=len(self.pocket_type_encoder)
            )

        else:
            pocket_atoms = [a for res in biopython_residues
                            for a in res.get_atoms()
                            if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
            pocket_coord = torch.tensor(np.array(
                [a.get_coord() for a in pocket_atoms]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[a.element.capitalize()]
                 for a in pocket_atoms], device=self.device)

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.pocket_type_encoder)
        )

        pocket_size = torch.tensor([len(pocket_coord)] * repeats,
                                   device=self.device, dtype=INT_TYPE)
        pocket_mask = torch.repeat_interleave(
            torch.arange(repeats, device=self.device, dtype=INT_TYPE),
            len(pocket_coord)
        )

        pocket = {
            'x': pocket_coord.repeat(repeats, 1),
            'one_hot': pocket_one_hot.repeat(repeats, 1),
            'size': pocket_size,
            'mask': pocket_mask
        }

        return pocket

    def generate_ligands(self, pdb_file, n_samples, pocket_ids=None,
                     ref_ligand=None, num_nodes_lig=None, sanitize=False,
                     largest_frag=False, relax_iter=0,
                     num_sampling_steps=None,
                     n_nodes_bias=0, n_nodes_min=0,
                     sigma_min_pos=None, sigma_max_pos=None,
                     sigma_min_feat=None, sigma_max_feat=None,
                     rho=None, dilated=None,
                     tau_start=None, tau_end=None,
                     # Inpainting / Repaint
                     ligand_inp=None,
                     lig_mask_fixed=None,
                     resamplings=1,
                     jump_length=1,              
                     repulsion_scale=0.0,        
                     repulsion_cutoff=1.2, 
                     ):
        """
        Generate ligands given a pocket using EDM sampling.

        Three execution paths:
          1. De-novo via sample_denovo (resamplings=1, no ligand_inp)
             Standard Heun sampler, fastest.
          2. De-novo via inpaint with all-zero mask
             (resamplings > 1, no ligand_inp)
             Routes through edm.inpaint() so resamplings apply to every
             denoising step even though no atoms are fixed.
          3. Inpainting of a partial ligand (ligand_inp + lig_mask_fixed provided)
             Fixed atoms are held to ground-truth at each step via Repaint.

        Args:
            pdb_file           : path to receptor PDB
            n_samples          : number of molecules to generate
            pocket_ids         : list of "<chain>:<resi>" strings (mutually
                                 exclusive with ref_ligand)
            ref_ligand         : "<chain>:<resi>" or SDF path for pocket def
            num_nodes_lig      : per-sample atom counts [n_samples]; sampled
                                 from learned distribution if None
            sanitize           : RDKit sanitize flag passed to process_molecule
            largest_frag       : keep only largest fragment
            relax_iter         : MMFF relaxation steps (0 = skip)
            num_sampling_steps : override Karras denoising steps
            n_nodes_bias       : additive bias on sampled node count
            n_nodes_min        : lower bound on node count
            sigma_*            : Karras schedule overrides
            ligand_inp         : dict {'x','one_hot','size','mask'} for the
                                 FULL already-batched ligand (all n_samples).
                                 If None and resamplings> 1,
                                 a zero-filled dummy is created automatically.
            lig_mask_fixed     : float tensor [N_total_lig], 1=fixed 0=generate.
                                 Must match ligand_inp atom ordering exactly.
            resamplings        : Repaint r — inner re-noise cycles per step

        Returns:
            list of RDKit Mol objects (length <= n_samples)
        """
        assert (pocket_ids is None) ^ (ref_ligand is None)

        self.model.eval()

        # ── Load PDB and prepare pocket ───────────────────────────────────
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]
        else:
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        pocket = self.prepare_pocket(residues, repeats=n_samples)

        pocket_for_model = {k: v.clone() for k, v in pocket.items()}

        common_schedule_kwargs = dict(
            sigma_min_pos=sigma_min_pos, sigma_max_pos=sigma_max_pos,
            sigma_min_feat=sigma_min_feat, sigma_max_feat=sigma_max_feat,
            rho=rho, dilated=dilated,
            tau_start=tau_start, tau_end=tau_end,
        )
        inpaint_schedule_kwargs  = dict(timesteps=num_sampling_steps,
                                        **common_schedule_kwargs)
        denovo_schedule_kwargs   = dict(num_sampling_steps=num_sampling_steps,
                                        **common_schedule_kwargs)

        use_inpaint = (ligand_inp is not None and lig_mask_fixed is not None) \
                      or (resamplings > 1)

        if use_inpaint:
            if num_nodes_lig is None:
                num_nodes_lig = self.model.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])
            num_nodes_lig = torch.clamp(
                num_nodes_lig + n_nodes_bias, min=n_nodes_min)

            if ligand_inp is None:
                batch_mask = utils.num_nodes_to_batch_mask(
                    n_samples, num_nodes_lig, self.device)
                n_total = len(batch_mask)

                ligand_batch = {
                    'x':       torch.zeros((n_total, self.x_dims),
                                           device=self.device, dtype=FLOAT_TYPE),
                    'one_hot': torch.zeros((n_total, self.atom_nf),
                                           device=self.device, dtype=FLOAT_TYPE),
                    'size':    num_nodes_lig,
                    'mask':    batch_mask,
                }
                mask_batch = torch.zeros(n_total,
                                         device=self.device, dtype=FLOAT_TYPE)
            else:

                ligand_batch = ligand_inp
                mask_batch   = lig_mask_fixed

            xh_lig_out, _, lig_mask_out, _ = self.model.inpaint(
                ligand=ligand_batch,
                pocket=pocket_for_model,
                lig_mask_fixed=mask_batch,
                resamplings=resamplings,
                jump_length=jump_length, 
                repulsion_scale=repulsion_scale,
                repulsion_cutoff=repulsion_cutoff,
                **inpaint_schedule_kwargs,
            )

            x_gen    = xh_lig_out[:, :self.model.n_dims]
            h_gen    = xh_lig_out[:, self.model.n_dims:]
            lig_mask = lig_mask_out.cpu()

        else:
            # Path 1: standard de-novo via sample_denovo (fastest)
            if num_nodes_lig is None:
                num_nodes_lig = self.model.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])
            num_nodes_lig = torch.clamp(
                num_nodes_lig + n_nodes_bias, min=n_nodes_min)

            x_gen, h_gen = self.model.sample_denovo(
                pocket_for_model, num_nodes_lig, **denovo_schedule_kwargs,
            )

        atom_type = h_gen.argmax(dim=1).detach().cpu()
        x         = x_gen.detach().cpu()

        if not use_inpaint:
            lig_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device).cpu()

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                        utils.batch_to_list(atom_type, lig_mask)):
            mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                add_hydrogens=False,
                                sanitize=sanitize,
                                relax_iter=relax_iter,
                                largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def configure_gradient_clipping(self, optimizer,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
            
    def _compute_atom_type_prior(self):
        """Compute atom type prior p(z) from dataset histogram."""
        atom_decoder = self.dataset_info['atom_decoder']
        atom_hist = self.dataset_info['atom_hist']
        
        counts = torch.tensor(
            [atom_hist[a] for a in atom_decoder], dtype=torch.float32
        )
        counts = counts + 1.0  # Laplace smoothing
        return counts / counts.sum()
    
class EDMWeightSchedule(nn.Module):
    def __init__(self, sigma_min, sigma_max, max_weight):
        super().__init__()
        self.max_weight = max_weight

        self.register_buffer('sigma_min', torch.tensor(sigma_min))
        self.register_buffer('sigma_max', torch.tensor(sigma_max))
        self.register_buffer('log_sigma_min', torch.tensor(math.log(sigma_min)))
        self.register_buffer('log_sigma_max', torch.tensor(math.log(sigma_max)))

    def forward(self, sigma):
        """
        Args:
            sigma: Tensor hoặc float
        """
        # 1. Xử lý input an toàn & đúng device
        if not isinstance(sigma, torch.Tensor):
            # Lấy device từ buffer đã đăng ký (luôn đúng theo model)
            sigma = torch.tensor(sigma, device=self.log_sigma_min.device)
        else:
            # Đảm bảo sigma cùng device với model (đề phòng input lạ)
            sigma = sigma.to(self.log_sigma_min.device)

        # 2. Giới hạn giá trị (Clamp)
        sigma_clamped = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        
        # 3. Tính log
        log_sigma = torch.log(sigma_clamped)
        
        # 4. Tính tỷ lệ (Ratio) trong không gian Log
        # ratio = (log(sigma) - log(min)) / (log(max) - log(min))
        ratio = (log_sigma - self.log_sigma_min) / (self.log_sigma_max - self.log_sigma_min)
        
        # 5. Weight decay: Max tại min noise, 0 tại max noise
        weight_factor = torch.clamp(1.0 - ratio, 0.0, 1.0)
        
        return self.max_weight * weight_factor
    
class ConfidenceMonitor:
    def __init__(self, outdir, sigma_max=80.0, sigma_min=0.002, rho=7):
        self.outdir = Path(outdir)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        
        self.history = {
            'sigma': [],
            'entropy': [],
            'max_prob': []
        }

    def update(self, z_h, sigma_val):
        with torch.no_grad():
            probs = F.softmax(z_h, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            max_p = probs.max(dim=-1)[0].mean().item()

            self.history['sigma'].append(sigma_val)
            self.history['entropy'].append(entropy)
            self.history['max_prob'].append(max_p)

    def calculate_tau(self, sigma_val):
        """
        Inverse of Karras EDM schedule:
            tau = (sigma^{1/rho} - sigma_max^{1/rho}) / (sigma_min^{1/rho} - sigma_max^{1/rho})
        
        NOTE: sigma_val is already in physical units (no sigma_data scaling).
        """
        inv_rho = 1.0 / self.rho
        term_val = sigma_val ** inv_rho
        term_max = self.sigma_max ** inv_rho
        term_min = self.sigma_min ** inv_rho
        
        denominator = term_min - term_max
        if abs(denominator) < 1e-9: 
            return 0.0
            
        tau = (term_val - term_max) / denominator
        return max(0.0, min(1.0, tau))

    def save_data(self, epoch):
        # Lưu cả ảnh và CSV
        self.save_plot(epoch)
        self.save_csv(epoch)

    def save_plot(self, epoch):
        sigmas = self.history['sigma']        
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Noise Level (Sigma)')
        ax1.set_ylabel('Entropy', color=color)
        ax1.plot(sigmas, self.history['entropy'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.invert_xaxis() 

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Max Probability', color=color)
        ax2.plot(sigmas, self.history['max_prob'], color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Confidence Trajectory - Epoch {epoch}")
        plt.savefig(self.outdir / f'entropy_epoch_{epoch}.png')
        plt.close(fig)

    def save_csv(self, epoch):
        csv_file = self.outdir / f'entropy_epoch_{epoch}.csv'
        
        # Tính toán Tau cho toàn bộ lịch sử
        taus = [self.calculate_tau(s) for s in self.history['sigma']]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Ghi header
            writer.writerow(['step', 'sigma', 'tau_calculated', 'entropy', 'max_prob'])
            
            # Ghi dữ liệu từng bước
            for i, (s, t, e, p) in enumerate(zip(self.history['sigma'], taus, self.history['entropy'], self.history['max_prob'])):
                writer.writerow([i, s, f"{t:.4f}", f"{e:.4f}", f"{p:.4f}"])
                
        print(f"Saved entropy data and calculations to {csv_file}")