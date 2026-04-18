from typing import Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import networkx as nx
from networkx.algorithms import isomorphism
from Bio.PDB.Polypeptide import is_aa


class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


#####


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        norm_type: float = 2.0) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in
         parameters]), norm_type)

    return total_norm


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    #with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m)

    # print(f'Wrote SDF file to {sdf_path}')


def residues_to_atoms(x, atom_encoder, representation='CA'):
    device = x.device
    num_classes = len(atom_encoder)
    
    if representation == 'backbone':
        idx_n = atom_encoder.get('N', 0)
        idx_c = atom_encoder.get('C', 0)
        idx_o = atom_encoder.get('O', 0)
        
        residue_pattern = torch.tensor([idx_n, idx_c, idx_c, idx_o], device=device)
        
        n_atoms = x.shape[-2]
        if n_atoms % 4 != 0:
            indices = torch.full((n_atoms,), idx_c, device=device, dtype=torch.long)
        else:
            n_residues = n_atoms // 4
            indices = residue_pattern.repeat(n_residues)
        
        one_hot = F.one_hot(indices, num_classes=num_classes) # [N, num_classes]
        
        if x.ndim > 2: 
            view_shape = [1] * (x.ndim - 2) + [n_atoms, num_classes]
            one_hot = one_hot.view(*view_shape).expand(*x.shape[:-1], num_classes)
            
    else:
        idx_c = atom_encoder.get('C', 0)
        one_hot = F.one_hot(
            torch.tensor(idx_c, device=device),
            num_classes=num_classes
        ).repeat(*x.shape[:-1], 1)

    return x, one_hot


def get_residue_with_resi(pdb_chain, resi):
    res = [x for x in pdb_chain.get_residues() if x.id[1] == resi]
    assert len(res) == 1
    return res[0]


def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):

    if ligand.endswith(".sdf"):
        # ligand as sdf file
        rdmol = Chem.SDMolSupplier(str(ligand))[0]
        ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        resi = None
    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ligand.split(':')
        ligand = get_residue_with_resi(pdb_model[chain], int(resi))
        ligand_coords = torch.from_numpy(
            np.array([a.get_coord() for a in ligand.get_atoms()]))

    pocket_residues = []
    for residue in pdb_model.get_residues():
        if residue.id[1] == resi:
            continue  # skip ligand itself

        res_coords = torch.from_numpy(
            np.array([a.get_coord() for a in residue.get_atoms()]))
        if is_aa(residue.get_resname(), standard=True) \
                and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
            pocket_residues.append(residue)

    return pocket_residues


def batch_to_list(data, batch_mask):
    # data_list = []
    # for i in torch.unique(batch_mask):
    #     data_list.append(data[batch_mask == i])
    # return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    for atom in rdmol.GetAtoms():
        # Add the atoms as nodes
        graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph


def calc_rmsd(mol_a, mol_b):
    """ Calculate RMSD of two molecules with unknown atom correspondence. """
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = isomorphism.GraphMatcher(
        graph_a, graph_b,
        node_match=lambda na, nb: na['atom_type'] == nb['atom_type'])

    isomorphisms = list(gm.isomorphisms_iter())
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
                        for i in range(mol_b.GetNumAtoms())]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array([conf_a.GetAtomPosition(i)
                             for i in range(mol_a.GetNumAtoms())])
        conf_b = mol_b.GetConformer()
        coords_b = np.array([conf_b.GetAtomPosition(mapping[i])
                             for i in range(mol_b.GetNumAtoms())])

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)


class AppendVirtualNodes:
    def __init__(self, max_ligand_size, atom_encoder, symbol):
        self.max_ligand_size = max_ligand_size
        self.atom_encoder = atom_encoder
        self.vidx = atom_encoder[symbol]

    def __call__(self, data):

        n_virt = self.max_ligand_size - data['num_lig_atoms']
        mu = data['lig_coords'].mean(0, keepdim=True)
        sigma = data['lig_coords'].std(0).max()
        virt_coords = torch.randn(n_virt, 3) * sigma + mu

        # insert virtual atom column
        one_hot = torch.cat((data['lig_one_hot'][:, :self.vidx],
                            torch.zeros(data['num_lig_atoms'])[:, None],
                            data['lig_one_hot'][:, self.vidx:]), dim=1)
        virt_one_hot = torch.zeros(n_virt, len(self.atom_encoder))
        virt_one_hot[:, self.vidx] = 1
        virt_mask = torch.ones(n_virt) * data['lig_mask'][0]

        data['lig_coords'] = torch.cat((data['lig_coords'], virt_coords))
        data['lig_one_hot'] = torch.cat((one_hot, virt_one_hot))
        data['num_lig_atoms'] = self.max_ligand_size
        data['lig_mask'] = torch.cat((data['lig_mask'], virt_mask))
        data['num_virtual_atoms'] = n_virt

        return data

import torch


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements the learning rate schedule defined AF3.

    A linear warmup is followed by a plateau at the maximum
    learning rate and then exponential decay. Note that the
    initial learning rate of the optimizer in question is
    ignored; use this class' base_lr parameter to specify
    the starting point of the warmup.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        base_lr: float = 0.0,
        max_lr: float = 1.8e-3,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        step_counts = {
            "warmup_no_steps": warmup_no_steps,
            "start_decay_after_n_steps": start_decay_after_n_steps,
        }

        for k, v in step_counts.items():
            if v < 0:
                msg = f"{k} must be nonnegative"
                raise ValueError(msg)

        if warmup_no_steps > start_decay_after_n_steps:
            msg = "warmup_no_steps must not exceed start_decay_after_n_steps"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        super().__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self) -> dict:
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            msg = (
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
            raise RuntimeError(msg)

        step_no = self.last_epoch

        if step_no <= self.warmup_no_steps:
            lr = self.base_lr + (step_no / self.warmup_no_steps) * self.max_lr
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exp = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor**exp)
        else:  # plateau
            lr = self.max_lr

        return [lr for group in self.optimizer.param_groups]