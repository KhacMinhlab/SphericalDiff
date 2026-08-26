import argparse
import warnings
from pathlib import Path
from time import time

import torch
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketEDM
from analysis.molecule_builder import process_molecule
import utils
import csv
import os

MAXITER = 10
MAXNTRIES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--test_list', type=Path, default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='Samples per batch (reduced from 120 for EDM Heun sampler)')
    parser.add_argument('--resamplings', type=int, default=1, 
                        help='Resampling Repaint step (mặc định: 1)')
    parser.add_argument('--jump_length', type=int, default=1,
                        help=("Repaint J: jump-length for the backward re-noising schedule. "),)
    parser.add_argument('--repulsion_scale', type=float, default=0.0,
                        help='Steric repulsion guidance strength (default: 0.0)')
    parser.add_argument('--repulsion_cutoff', type=float, default=1.2,
                        help='Distance cutoff for repulsion (default: 1.2)')
    parser.add_argument('--num_sampling_steps', type=int, default=None,
                        help='Override EDM denoising steps (default: use training config)')
    

    parser.add_argument('--fix_n_nodes', action='store_true')
    parser.add_argument('--n_nodes_bias', type=int, default=0)
    parser.add_argument('--n_nodes_min', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--sigma_min_pos', type=float, default=None,
                    help='Override sigma_min for positions')
    
    # Karras noise schedule overrides
    parser.add_argument('--sigma_max_pos', type=float, default=None,
                        help='Override sigma_max for positions')
    parser.add_argument('--sigma_min_feat', type=float, default=None,
                        help='Override sigma_min for features')
    parser.add_argument('--sigma_max_feat', type=float, default=None,
                        help='Override sigma_max for features')
    parser.add_argument('--rho', type=float, default=None,
                        help='Override rho exponent for Karras schedule')
    parser.add_argument('--dilated', type=lambda x: x.lower() == 'true',
                        default=None,
                        help='Override dilated schedule (true/false)')
    parser.add_argument('--tau_start', type=float, default=None,
                        help='Override dilation region start')
    parser.add_argument('--tau_end', type=float, default=None,
                        help='Override dilation region end')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.outdir.mkdir(exist_ok=args.skip_existing)
    raw_sdf_dir = Path(args.outdir, 'raw')
    raw_sdf_dir.mkdir(exist_ok=args.skip_existing)
    processed_sdf_dir = Path(args.outdir, 'processed')
    processed_sdf_dir.mkdir(exist_ok=args.skip_existing)
    times_dir = Path(args.outdir, 'pocket_times')
    times_dir.mkdir(exist_ok=args.skip_existing)

    # Load model
    model = LigandPocketEDM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)
    ligand_metrics = model.ligand_metrics

    test_files = list(args.test_dir.glob('[!.]*.sdf'))
    if args.test_list is not None:
        with open(args.test_list, 'r') as f:
            test_list = set(f.read().split(','))
        test_files = [x for x in test_files if x.stem in test_list]

    pocket_stats    = []
    time_per_pocket = {}

    pbar = tqdm(test_files)
    for sdf_file in pbar:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split('_')
        pdb_file = Path(sdf_file.parent, f"{pdb_name}_{pocket_id}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f'{ligand_name}_gen.sdf')
        sdf_out_file_processed = Path(processed_sdf_dir,
                                      f'{ligand_name}_gen.sdf')
        time_file = Path(times_dir, f'{ligand_name}.txt')

        if not os.path.exists(pdb_file):
            # Note: For CrossDocked, the corresponding PDB filepath is found at a different location
            pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
            assert os.path.exists(pdb_file), f"CrossDocked PDB file {pdb_file} must be available to test with CrossDocked."

        if args.skip_existing and time_file.exists() \
                and sdf_out_file_processed.exists() \
                and sdf_out_file_raw.exists():

            with open(time_file, 'r') as f:
                time_per_pocket[str(sdf_file)] = float(f.read().split()[1])

            continue

        for n_try in range(MAXNTRIES):

            try:
                t_pocket_start = time()

                with open(txt_file, 'r') as f:
                    resi_list = f.read().split()

                if args.fix_n_nodes:
                    # some ligands could not be read with sanitize=True
                    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
                    num_nodes_lig = suppl[0].GetNumAtoms()
                else:
                    num_nodes_lig = None

                all_molecules = []
                valid_molecules = []
                processed_molecules = []
                all_mols_for_metrics = []
                iter = 0
                n_generated = 0
                n_process_pass = 0
                while len(valid_molecules) < args.n_samples:
                    iter += 1
                    if iter > MAXITER:
                        raise RuntimeError('Maximum number of iterations has been exceeded.')

                    num_nodes_lig_inflated = None if num_nodes_lig is None else \
                        torch.ones(args.batch_size, dtype=int) * num_nodes_lig

                    mols_batch = model.generate_ligands(
                        pdb_file, args.batch_size, resi_list,
                        num_nodes_lig=num_nodes_lig_inflated,
                        sanitize=False,
                        largest_frag=False, relax_iter=0,
                        num_sampling_steps=args.num_sampling_steps,
                        n_nodes_bias=args.n_nodes_bias,
                        n_nodes_min=args.n_nodes_min,
                        sigma_min_pos=args.sigma_min_pos,
                        sigma_max_pos=args.sigma_max_pos,
                        sigma_min_feat=args.sigma_min_feat,
                        sigma_max_feat=args.sigma_max_feat,
                        rho=args.rho,
                        dilated=args.dilated,
                        tau_start=args.tau_start,
                        tau_end=args.tau_end,
                        resamplings=args.resamplings,
                        jump_length=args.jump_length,
                        repulsion_scale=args.repulsion_scale,
                        repulsion_cutoff=args.repulsion_cutoff)

                    all_molecules.extend(mols_batch)
                    all_mols_for_metrics.extend(mols_batch)
                    # Filter to find valid molecules
                    mols_batch_processed = [
                        process_molecule(m, sanitize=args.sanitize,
                                         relax_iter=(200 if args.relax else 0),
                                         largest_frag=not args.all_frags)
                        for m in mols_batch
                    ]
                    processed_molecules.extend(mols_batch_processed)
                    valid_mols_batch = [m for m in mols_batch_processed if m is not None]

                    n_generated += args.batch_size
                    n_process_pass += len(valid_mols_batch)
                    valid_molecules.extend(valid_mols_batch)

                # Remove excess molecules from list
                valid_molecules = valid_molecules[:args.n_samples]

                # Reorder raw files
                all_molecules = \
                    [all_molecules[i] for i, m in enumerate(processed_molecules)
                     if m is not None] + \
                    [all_molecules[i] for i, m in enumerate(processed_molecules)
                     if m is None]

                # Write SDF files
                utils.write_sdf_file(sdf_out_file_raw, all_molecules)
                utils.write_sdf_file(sdf_out_file_processed, valid_molecules)

                # Time the sampling process
                time_per_pocket[str(sdf_file)] = time() - t_pocket_start
                with open(time_file, 'w') as f:
                    f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

                valid_mols_m, validity = \
                    ligand_metrics.compute_validity(all_mols_for_metrics)

                connected_mols, connectivity, connected_smiles  = \
                    ligand_metrics.compute_connectivity(valid_mols_m)
                
                validity_pct     = validity     * 100
                connectivity_pct = connectivity * 100
                process_pct      = n_process_pass / n_generated * 100 if n_generated else 0.0
                
                pocket_stats.append({
                    'pocket':           ligand_name,
                    'n_generated':      n_generated,
                    'n_valid':          len(valid_mols_m),
                    'n_connected':      len(connected_mols),
                    'n_process_pass':   n_process_pass,
                    'validity_pct':     round(validity_pct,     2),
                    'connectivity_pct': round(connectivity_pct, 2),
                    'process_pct':      round(process_pct,      2),
                    'time_sec':         round(time_per_pocket[str(sdf_file)], 3),
                })
                pbar.set_description(
                    f'{ligand_name} | '
                    f'valid={validity_pct:.1f}% '
                    f'conn={connectivity_pct:.1f}% '
                    f'proc={process_pct:.1f}% '
                    f't={time_per_pocket[str(sdf_file)]:.1f}s'
                )

                break  # no more tries needed

            except (RuntimeError, ValueError) as e:
                if n_try >= MAXNTRIES - 1:
                    raise RuntimeError("Maximum number of retries exceeded")
                warnings.warn(f"Attempt {n_try + 1}/{MAXNTRIES} failed with "
                              f"error: '{e}'. Trying again...")

    with open(Path(args.outdir, 'pocket_times.txt'), 'w') as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(f"Time per pocket: {times_arr.mean():.3f} \u00b1 "
          f"{times_arr.std(unbiased=False):.2f}")
    
    if not pocket_stats:
        print("No pockets processed.")
    else:
        # -- Per-pocket CSV ------------------------------------------------
        csv_path = Path(args.outdir, 'metrics_per_pocket.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(pocket_stats[0].keys()))
            writer.writeheader()
            writer.writerows(pocket_stats)
        print(f"\nPer-pocket metrics -> {csv_path}")

        # -- Aggregate summary ---------------------------------------------
        total_gen  = sum(s['n_generated']    for s in pocket_stats)
        total_val  = sum(s['n_valid']        for s in pocket_stats)
        total_conn = sum(s['n_connected']    for s in pocket_stats)
        total_proc = sum(s['n_process_pass'] for s in pocket_stats)
        mean_time  = sum(s['time_sec']       for s in pocket_stats) / len(pocket_stats)

        summary_lines = [
            "=" * 58,
            "AGGREGATE METRICS",
            "=" * 58,
            f"  Pockets evaluated   : {len(pocket_stats)}",
            f"  Total generated     : {total_gen}",
            f"",
            f"  Validity            : {total_val  / total_gen * 100:.2f}%",
            f"    [BasicMolecularMetrics.compute_validity]",
            f"    SanitizeMol on raw mols, catches ValueError",
            f"",
            f"  Connectivity        : {total_conn / total_val * 100:.2f}% (of valid)",
            f"    [BasicMolecularMetrics.compute_connectivity]",
            f"    largest_fragment / total_atoms >= {ligand_metrics.connectivity_thresh}",
            f"",
            f"  Process pass rate   : {total_proc / total_gen * 100:.2f}%",
            f"    [process_molecule] sanitize={args.sanitize}, relax={args.relax}",
            f"",
            f"  Mean time/pocket    : {mean_time:.2f} sec",
            f"",
            f"  resamplings (r)     : {args.resamplings}",
            f"  jump_length (J)     : {args.jump_length}",
            "=" * 58,
        ]

        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        summary_path = Path(args.outdir, 'metrics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text + "\n")
        print(f"Summary -> {summary_path}")