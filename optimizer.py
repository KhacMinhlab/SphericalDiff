import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
import pandas as pd
import random
from torch_scatter import scatter_mean
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketEDM
from constants import FLOAT_TYPE, INT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties


def prepare_from_sdf_files(sdf_files, atom_encoder):

    ligand_coords = []
    atom_one_hot = []
    for file in sdf_files:
        rdmol = Chem.SDMolSupplier(str(file), sanitize=False)[0]
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()])
        atom_one_hot.append(
            F.one_hot(types, num_classes=len(atom_encoder))
        )

    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligands_from_mols(mols, atom_encoder, device='cpu'):

    ligand_coords = []
    atom_one_hots = []
    masks = []
    sizes = []
    for i, mol in enumerate(mols):
        coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=FLOAT_TYPE)
        types = torch.tensor([atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()], dtype=INT_TYPE)
        one_hot = F.one_hot(types, num_classes=len(atom_encoder))
        mask = torch.ones(len(types), dtype=INT_TYPE) * i
        ligand_coords.append(coord)
        atom_one_hots.append(one_hot)
        masks.append(mask)
        sizes.append(len(types))

    ligand = {
        'x': torch.cat(ligand_coords, dim=0).to(device),
        'one_hot': torch.cat(atom_one_hots, dim=0).to(device),
        'size': torch.tensor(sizes, dtype=INT_TYPE).to(device),
        'mask': torch.cat(masks, dim=0).to(device),
    }

    return ligand


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):

    coord = torch.tensor(np.array([a.get_coord()
                                   for a in biopython_atoms]), dtype=FLOAT_TYPE)
    types = torch.tensor([atom_encoder[a.element.capitalize()]
                          for a in biopython_atoms])
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))

    return coord, one_hot


def prepare_substructure(ref_ligand, fix_atoms, pdb_model):

    if fix_atoms[0].endswith(".sdf"):
        # ligand as sdf file
        coord, one_hot = prepare_from_sdf_files(fix_atoms, model.lig_type_encoder)

    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ref_ligand.split(':')
        ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))
        fixed_atoms = [a for a in ligand.get_atoms() if a.get_name() in set(fix_atoms)]
        coord, one_hot = prepare_ligand_from_pdb(fixed_atoms, model.lig_type_encoder)

    return coord, one_hot


def diversify_ligands(model, pocket, mols, timesteps,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0):
    """
    Diversify ligands for a specified pocket.
    
    Parameters:
        model: The model instance used for diversification.
        pocket: The pocket information including coordinates and types.
        mols: List of RDKit molecule objects to be diversified.
        timesteps: Number of denoising steps to apply during diversification.
        sanitize: If True, performs molecule sanitization post-generation (default: False).
        largest_frag: If True, only the largest fragment of the generated molecule is returned (default: False).
        relax_iter: Number of iterations for force field relaxation of the generated molecules (default: 0).
    
    Returns:
        A list of diversified RDKit molecule objects.
    """

    ligand = prepare_ligands_from_mols(mols, model.lig_type_encoder, device=model.device)

    pocket_mask = pocket['mask']
    lig_mask = ligand['mask']

    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    out_lig, out_pocket, _, _ = model.model.diversify(ligand, pocket, noising_steps=timesteps)

    # Move generated molecule back to the original pocket position
    pocket_com_after = scatter_mean(out_pocket[:, :model.x_dims], pocket_mask, dim=0)

    out_pocket[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[pocket_mask]
    out_lig[:, :model.x_dims] += \
        (pocket_com_before - pocket_com_after)[lig_mask]

    # Build mol objects
    x = out_lig[:, :model.x_dims].detach().cpu()
    atom_type = out_lig[:, model.x_dims:].argmax(1).detach().cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                      utils.batch_to_list(atom_type, lig_mask)):

        mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
        mol = process_molecule(mol,
                               add_hydrogens=False,
                               sanitize=sanitize,
                               relax_iter=relax_iter,
                               largest_frag=largest_frag)
        if mol is not None:
            molecules.append(mol)

    return molecules
import torch

def diversify_edm(model_wrapper, pocket, mols, steps_to_run=10, sanitize=True, relax_iter=0):
    """
    Hàm Diversify dành riêng cho kiến trúc EDM.
    Cơ chế: Thêm nhiễu vào ligand hiện tại ở mức sigma tương ứng, sau đó denoise lại.
    """
    # 1. Chuẩn bị dữ liệu
    device = model_wrapper.device
    edm_model = model_wrapper.model # Lấy model EDM từ wrapper Lightning
    
    # Chuyển list RDKit Mols thành Tensor Batch
    ligand = prepare_ligands_from_mols(mols, model_wrapper.lig_type_encoder, device=device)
    
    # 2. Tính toán mức Sigma bắt đầu
    # EDM loop chạy từ 0 (Noise lớn nhất) -> N (Noise nhỏ nhất)
    # Nếu muốn thay đổi nhỏ (diversify), ta bắt đầu từ gần cuối.
    # steps_to_run: Số bước denoising sẽ chạy (ví dụ 10 bước cuối)
    
    total_steps = edm_model.num_sampling_steps
    # Đảm bảo không vượt quá số bước tối đa
    steps_to_run = min(steps_to_run, total_steps) 
    
    # Chỉ số bắt đầu vòng lặp (càng lớn thì sigma càng nhỏ -> thay đổi càng ít)
    start_idx = total_steps - steps_to_run 
    
    # Lấy lịch trình sigma
    sigmas = edm_model.get_sigma_schedule(total_steps)
    start_sigma = sigmas[start_idx] # Mức nhiễu tại điểm bắt đầu
    
    # 3. Chuẩn hóa & Thêm nhiễu (Noising)
    # Lưu ý: Cần normalize pocket và ligand trước
    ligand_norm, pocket_norm = edm_model.normalize(ligand=ligand, pocket=pocket)
    
    xh_pocket = torch.cat([pocket_norm['x'], pocket_norm['one_hot']], dim=1)
    
    x_lig = ligand_norm['x']
    h_lig = ligand_norm['one_hot'].float()
    
    # Tạo nhiễu
    z_x = torch.randn_like(x_lig) * start_sigma + x_lig
    z_h = torch.randn_like(h_lig) * start_sigma + h_lig # Với EDM continuous, ta noise cả feature
    
    # Center gravity (quan trọng cho EDM)
    z_x, _ = edm_model.remove_mean_batch(z_x, xh_pocket[:, :3], ligand['mask'], pocket['mask'])

    # 4. Chạy vòng lặp Denoising (Sampling Loop)
    # Copy logic từ sample_denovo nhưng bắt đầu từ start_idx
    batch_size = len(ligand['size'])
    lig_mask = ligand['mask']
    
    # --- BẮT ĐẦU VÒNG LẶP EDM ---
    with torch.no_grad():
        for i in range(start_idx, total_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # (Phần này copy logic Heun Sampler từ edm.py)
            s_vec = torch.full((batch_size,), sigma_cur, device=device)
            s_mapped = s_vec[lig_mask].unsqueeze(1)
            
            # Coefficients
            cin_x = edm_model.c_in(s_mapped, edm_model.sigma_data_pos)
            cout_x = edm_model.c_out(s_mapped, edm_model.sigma_data_pos)
            cskip_x = edm_model.c_skip(s_mapped, edm_model.sigma_data_pos)
            
            cin_h = edm_model.c_in(s_mapped, edm_model.sigma_data_feat)
            cout_h = edm_model.c_out(s_mapped, edm_model.sigma_data_feat)
            cskip_h = edm_model.c_skip(s_mapped, edm_model.sigma_data_feat)
            
            c_noise = (s_vec / edm_model.sigma_data_pos).log() * 0.25
            
            # Predict
            net_in = torch.cat([z_x * cin_x, z_h * cin_h], dim=-1)
            raw_out, _, _ = edm_model.dynamics(net_in, xh_pocket, c_noise, lig_mask, pocket['mask'])
            out_x, out_h = raw_out[:, :3], raw_out[:, 3:]
            
            d_x = cskip_x * z_x + cout_x * out_x
            d_h = cskip_h * z_h + cout_h * out_h
            
            grad_x = (z_x - d_x) / s_mapped
            grad_h = (z_h - d_h) / s_mapped
            dt = sigma_next - sigma_cur
            
            next_x = z_x + grad_x * dt
            next_h = z_h + grad_h * dt
            
            # Heun Correction (2nd order)
            if sigma_next > 0:
                s_next_vec = torch.full((batch_size,), sigma_next, device=device)
                s_next_mapped = s_next_vec[lig_mask].unsqueeze(1)
                
                cin_x_2 = edm_model.c_in(s_next_mapped, edm_model.sigma_data_pos)
                cin_h_2 = edm_model.c_in(s_next_mapped, edm_model.sigma_data_feat)
                c_noise_2 = (s_next_vec / edm_model.sigma_data_pos).log() * 0.25
                
                net_in_2 = torch.cat([next_x * cin_x_2, next_h * cin_h_2], dim=-1)
                raw_out_2, _, _ = edm_model.dynamics(net_in_2, xh_pocket, c_noise_2, lig_mask, pocket['mask'])
                out_x_2, out_h_2 = raw_out_2[:, :3], raw_out_2[:, 3:]
                
                d_x_2 = edm_model.c_skip(s_next_mapped, edm_model.sigma_data_pos) * next_x + \
                        edm_model.c_out(s_next_mapped, edm_model.sigma_data_pos) * out_x_2
                d_h_2 = edm_model.c_skip(s_next_mapped, edm_model.sigma_data_feat) * next_h + \
                        edm_model.c_out(s_next_mapped, edm_model.sigma_data_feat) * out_h_2
                
                grad_x_2 = (next_x - d_x_2) / s_next_mapped
                grad_h_2 = (next_h - d_h_2) / s_next_mapped
                
                next_x = z_x + (grad_x + grad_x_2) * 0.5 * dt
                next_h = z_h + (grad_h + grad_h_2) * 0.5 * dt
            
            z_x = next_x
            z_h = next_h
            z_x, _ = edm_model.remove_mean_batch(z_x, xh_pocket[:, :3], lig_mask, pocket['mask'])

    # 5. Giải mã (Decoding)
    x_final, _ = edm_model.unnormalize(z_x, torch.zeros_like(z_h))
    
    # H (Atom Types): Argmax để lấy loại nguyên tử
    # Lưu ý: Cần unnormalize h trước nếu h cũng bị scale (trong config của bạn h scale=4)
    _, h_final_cont = edm_model.unnormalize(torch.zeros_like(z_x), z_h)
    atom_type = h_final_cont.argmax(1).detach().cpu()
    x_final = x_final.detach().cpu()
    lig_mask = lig_mask.cpu()
    
    # 6. Reconstruct Molecules (Copy logic cũ)
    # Lưu ý: Cần xử lý việc dời tọa độ (Back-mapping) giống hàm diversify cũ
    # Nhưng ở đây ta trả về raw molecules list, việc dời tọa độ nên làm bên ngoài hoặc tích hợp luôn
    
    # Lấy lại tâm pocket gốc để dời về
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0).cpu()
    # Tâm pocket hiện tại (trong model là 0, nhưng ta cần tham chiếu chuẩn)
    # Vì edm hoạt động ở zero-center, ta cộng lại tâm gốc vào kết quả
    
    # Logic dời tọa độ đơn giản:
    # Model EDM trả về tọa độ tại (0,0,0). Ta cộng thêm Pocket Center ban đầu.
    # (Khác với DDPM cũ phải tính delta giữa 2 lần)
    
    molecules_out = []
    
    # Tách batch
    x_list = utils.batch_to_list(x_final, lig_mask)
    h_list = utils.batch_to_list(atom_type, lig_mask)
    
    for i, (coords, atoms) in enumerate(zip(x_list, h_list)):
        # Cộng lại tâm Pocket
        coords = coords + pocket_com_before[i]
        
        mol_pc = (coords, atoms)
        mol = build_molecule(*mol_pc, model_wrapper.dataset_info, add_coords=True)
        mol = process_molecule(mol,
                               add_hydrogens=False,
                               sanitize=sanitize,
                               relax_iter=relax_iter,
                               largest_frag=True) # Luôn lấy mảnh lớn nhất
        if mol is not None:
            molecules_out.append(mol)
        else:
            # Fallback nếu lỗi, trả về mol cũ hoặc None
             pass
             
    return molecules_out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_linked_mols.sdf')
    parser.add_argument('--objective', type=str, default='sa', choices={'qed', 'sa'})
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--evolution_steps', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=7)
    parser.add_argument('--outfile', type=Path, default='output.sdf')
    parser.add_argument('--relax', action='store_true')


    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    population_size = args.population_size
    evolution_steps = args.evolution_steps
    top_k = args.top_k

    # Load model
    model = LigandPocketEDM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Prepare ligand + pocket
    # Load PDB
    pdb_model = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
    # Define pocket based on reference ligand
    residues = utils.get_pocket_from_ligand(pdb_model, args.ref_ligand)
    pocket = model.prepare_pocket(residues, repeats=population_size)


    if args.objective == 'qed':
        objective_function = MoleculeProperties().calculate_qed
    elif args.objective == 'sa':
        objective_function = MoleculeProperties().calculate_sa
    else:
        ### IMPLEMENT YOUR OWN OBJECTIVE
        ### FUNCTIONS HERE 
        raise ValueError(f"Objective function {args.objective} not recognized.")

    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]

    # Store molecules in history dataframe 
    buffer = pd.DataFrame(columns=['generation', 'score', 'fate' 'mol', 'smiles'])

    # Population initialization
    new_row = pd.DataFrame([{
        'generation': 0,
        'score': objective_function(ref_mol),
        'fate': 'initial', 
        'mol': ref_mol,
        'smiles': Chem.MolToSmiles(ref_mol)
    }])
    # Dùng concat để nối
    buffer = pd.concat([buffer, new_row], ignore_index=True)

    for generation_idx in range(evolution_steps):

        if generation_idx == 0:
            molecules = buffer['mol'].tolist() * population_size
        else:
            # Select top k molecules from previous generation
            previous_gen = buffer[buffer['generation'] == generation_idx]
            top_k_molecules = previous_gen.nlargest(top_k, 'score')['mol'].tolist()
            molecules = top_k_molecules * (population_size // top_k)

            # Update the fate of selected top k molecules in the buffer
            buffer.loc[buffer['generation'] == generation_idx, 'fate'] = 'survived'

            # Ensure the right number of molecules
            if len(molecules) < population_size:
                molecules += [random.choice(molecules) for _ in range(population_size - len(molecules))]


        # Diversify molecules
        assert len(molecules) == population_size, f"Wrong number of molecules: {len(molecules)} when it should be {population_size}"
        print(f"Generation {generation_idx}, mean score: {np.mean([objective_function(mol) for mol in molecules])}")
        molecules = diversify_edm(model, 
                                pocket, 
                                molecules, 
                                steps_to_run=args.timesteps, 
                                sanitize=True, 
                                relax_iter=(200 if args.relax else 0))
                
        
        # Evaluate and save molecules
        for mol in molecules:
            row_data = {
                'generation': generation_idx + 1,
                'score': objective_function(mol),
                'fate': 'purged',
                'mol': mol,
                'smiles': Chem.MolToSmiles(mol)
            }
            buffer = pd.concat([buffer, pd.DataFrame([row_data])], ignore_index=True)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
    # Save buffer
    buffer.drop(columns=['mol'])
    buffer.to_csv(args.outfile.with_suffix('.csv'))