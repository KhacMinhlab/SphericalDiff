import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketEDM
from analysis.molecule_builder import process_molecule
from constants import FLOAT_TYPE, INT_TYPE
import utils

try:
    from analysis.constrained_relax import process_molecule_inpaint
    _HAS_CONSTRAINED_RELAX = True
except ImportError:
    _HAS_CONSTRAINED_RELAX = False
    import warnings
    warnings.warn(
        "constrained_relax.py not found — falling back to standard "
        "process_molecule for inpainting.  Place constrained_relax.py "
        "in your project root to enable scaffold-pinned MMFF relaxation."
    )


def prepare_from_sdf_files(sdf_files: list, atom_encoder: dict):
    ligand_coords = []
    atom_one_hot  = []

    for file in sdf_files:
        supplier = Chem.SDMolSupplier(str(file), sanitize=False)
        n_read = 0
        for idx, rdmol in enumerate(supplier):
            if rdmol is None:
                print(f"[warn]  Skipping unreadable molecule block {idx} "
                      f"in {file}")
                continue
            rdmol = Chem.RemoveHs(rdmol)
            ligand_coords.append(
                torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
            )
            types = torch.tensor(
                [atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()]
            )
            atom_one_hot.append(
                F.one_hot(types, num_classes=len(atom_encoder))
            )
            n_read += 1

        if n_read == 0:
            raise ValueError(
                f"Could not read any molecule from SDF: {file}"
            )
        print(f"[info]  Read {n_read} molecule block(s) from {file}")

    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder: dict):
    """Extract coordinates and one-hot from BioPython Atom objects."""
    biopython_atoms = [a for a in biopython_atoms if a.element.capitalize() != 'H']
    coord = torch.tensor(
        np.array([a.get_coord() for a in biopython_atoms]), dtype=FLOAT_TYPE
    )
    types = torch.tensor(
        [atom_encoder[a.element.capitalize()] for a in biopython_atoms]
    )
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))
    return coord, one_hot


def prepare_substructure(ref_ligand: str, fix_atoms: list,
                         pdb_model, atom_encoder: dict):
    """Return (coords, one_hot) of the atoms to be held fixed."""
    if fix_atoms[0].endswith(".sdf"):
        coord, one_hot = prepare_from_sdf_files(fix_atoms, atom_encoder)
    else:
        chain, resi = ref_ligand.split(":")
        ligand      = utils.get_residue_with_resi(pdb_model[chain], int(resi))
        fixed_atoms = [a for a in ligand.get_atoms()
                       if a.get_name() in set(fix_atoms)]
        if not fixed_atoms:
            raise ValueError(
                f"None of the requested fix_atoms {fix_atoms} were found "
                f"in residue {ref_ligand}."
            )
        coord, one_hot = prepare_ligand_from_pdb(fixed_atoms, atom_encoder)

    return coord, one_hot


def assess_and_route_molecules(
    p1_valid: List[Chem.Mol],
    min_heavy_atoms: int = 15,
    sanitize: bool = False,
    n_fixed: int = 0,
) -> Tuple[List[dict], List[Chem.Mol], int]:
    """
    Assess Phase-1 molecules and route them for Phase-2 refinement.

    Strategy (no linking):
        1. Multi-fragment molecules → extract the fragment holding the fixed
           scaffold (atom 0) when inpainting (n_fixed > 0); otherwise the
           largest fragment
        2. Single fragment with < min_heavy_atoms → Group B (grow)
        3. Single fragment with ≥ min_heavy_atoms → Group C (pass)

    Returns
    -------
    group_b   : list of dicts for growing  (each has 'mol', 'heavy_atoms', 'min_added')
    group_c   : list of Mol that passed directly
    n_trimmed : count of molecules that had fragments trimmed
    """
    group_b: List[dict]     = []
    group_c: List[Chem.Mol] = []
    n_trimmed = 0

    for mol in p1_valid:
        try:
            frag_indices = Chem.GetMolFrags(mol, asMols=False)
            frags        = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            num_frags    = len(frags)
        except Exception:
            continue  # skip molecules that RDKit cannot analyse

        # ── Step 1: If multi-fragment, keep the fragment holding the fixed
        # scaffold (atom 0) when inpainting; otherwise keep the largest ──
        if num_frags > 1:
            if n_fixed > 0:
                mol = next(
                    (f for f, idx in zip(frags, frag_indices) if 0 in idx),
                    max(frags, key=lambda m: m.GetNumAtoms()),
                )
            else:
                mol = max(frags, key=lambda m: m.GetNumAtoms())
            if sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    continue
            n_trimmed += 1

        # ── Step 2: Route by size ─────────────────────────────────────
        heavy_atoms = mol.GetNumHeavyAtoms()

        if heavy_atoms < min_heavy_atoms:
            # Group B – Fragment Growing
            group_b.append({
                'mol':         mol,
                'heavy_atoms': heavy_atoms,
                'min_added':   1,
            })
        else:
            # Group C – Pass (large enough single fragment)
            group_c.append(mol)

    return group_b, group_c, n_trimmed



def sample_target_sizes(
    model: LigandPocketEDM,
    pocket_size: torch.Tensor,
    entries: List[dict],
    max_resample_attempts: int = 200,
) -> List[int]:
    size_dist = model.model.size_distribution
    add_n_list: List[int] = []

    for entry in entries:
        n_fixed    = entry['heavy_atoms']
        min_added  = entry['min_added']
        min_target = n_fixed + min_added

        add_n = None
        for _ in range(max_resample_attempts):
            n_target = size_dist.sample_conditional(
                n1=None, n2=pocket_size
            ).item()

            if n_target >= min_target:
                add_n = n_target - n_fixed
                break

        if add_n is None:
            add_n = min_added
            print(
                f"[warn]  size_distribution never sampled N_target >= "
                f"{min_target} for molecule with {n_fixed} atoms after "
                f"{max_resample_attempts} attempts; falling back to "
                f"add_n_nodes = {add_n}"
            )

        add_n_list.append(add_n)

    return add_n_list

def _mol_to_tensors(
    mol: Chem.Mol,
    atom_encoder: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (coords ``[N, 3]``, one_hot ``[N, n_types]``) from an RDKit Mol
    that already carries a 3D conformer.
    """
    conf   = mol.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=FLOAT_TYPE)

    n_types = len(atom_encoder)
    type_indices = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        type_indices.append(atom_encoder.get(sym, 0))   # fallback → Carbon

    types   = torch.tensor(type_indices, dtype=INT_TYPE)
    one_hot = F.one_hot(types, num_classes=n_types).to(FLOAT_TYPE)

    return coords, one_hot


def prepare_inpaint_batch(
    entries: List[dict],
    add_n_list: List[int],
    atom_encoder: dict,
    device: str = "cpu",
) -> Tuple[dict, torch.Tensor]:

    n_types = len(atom_encoder)

    all_x:       List[torch.Tensor] = []
    all_one_hot: List[torch.Tensor] = []
    all_mask:    List[torch.Tensor] = []
    all_fixed:   List[torch.Tensor] = []
    sizes:       List[int]          = []

    for batch_idx, (entry, add_n) in enumerate(zip(entries, add_n_list)):
        mol     = entry['mol']
        coords, one_hot = _mol_to_tensors(mol, atom_encoder)
        n_fixed = coords.shape[0]
        n_total = n_fixed + add_n

        # Coordinates: [N_fixed real coords | add_n zeros]
        x_padded = torch.zeros(n_total, 3, dtype=FLOAT_TYPE)
        x_padded[:n_fixed] = coords

        # One-hot features: [N_fixed real types | add_n zeros]
        h_padded = torch.zeros(n_total, n_types, dtype=FLOAT_TYPE)
        h_padded[:n_fixed] = one_hot

        # Batch-index mask (integer identifying which sample each atom belongs to)
        mask_i = torch.full((n_total,), batch_idx, dtype=INT_TYPE)

        # Fixed-atom mask: 1.0 = hold, 0.0 = generate
        fixed_i = torch.zeros(n_total, dtype=FLOAT_TYPE)
        fixed_i[:n_fixed] = 1.0

        all_x.append(x_padded)
        all_one_hot.append(h_padded)
        all_mask.append(mask_i)
        all_fixed.append(fixed_i)
        sizes.append(n_total)

    ligand_inp = {
        'x':       torch.cat(all_x,      dim=0).to(device),
        'one_hot': torch.cat(all_one_hot, dim=0).to(device),
        'size':    torch.tensor(sizes, dtype=INT_TYPE, device=device),
        'mask':    torch.cat(all_mask,    dim=0).to(device),
    }
    lig_mask_fixed = torch.cat(all_fixed, dim=0).to(device)

    return ligand_inp, lig_mask_fixed

def run_refinement(
    model: LigandPocketEDM,
    entries: List[dict],
    add_n_list: List[int],
    pdb_file: str,
    ref_ligand: str,
    pocket_ids: Optional[list],
    device: str,
    inpaint_resamplings: int = 5,
    jump_length: int = 1,                       # ← NEW
    repulsion_scale: float = 0.0,               # ← NEW
    repulsion_cutoff: float = 1.2,              # ← NEW
    num_sampling_steps: int = 40,
    sanitize: bool = False,
    relax_iter: int = 0,
    largest_frag: bool = False,
    karras_kwargs: Optional[dict] = None,
) -> List[Chem.Mol]:

    if not entries:
        return []

    if karras_kwargs is None:
        karras_kwargs = {}

    atom_encoder = model.lig_type_encoder

    # §3 – Build inpaint tensors
    ligand_inp, lig_mask_fixed = prepare_inpaint_batch(
        entries, add_n_list, atom_encoder, device=device
    )

    n_samples = len(entries)

    # Pass num_nodes_lig explicitly so that generate_ligands does NOT
    # re-sample sizes from its distribution (we already decided them).
    num_nodes_lig = ligand_inp['size'].clone()

    # Determine pocket reference (mutually exclusive args for generate_ligands)
    pocket_ref = ref_ligand if Path(ref_ligand).is_file() else None
    pocket_id_list = [ref_ligand] if ":" in ref_ligand else None

    if pocket_ids is not None:
        pocket_id_list = pocket_ids
        pocket_ref = None

    # §4 – Execute inpainting via generate_ligands
    repaired_mols = model.generate_ligands(
        pdb_file           = pdb_file,
        n_samples          = n_samples,
        ref_ligand         = pocket_ref,
        pocket_ids         = pocket_id_list,
        num_nodes_lig      = num_nodes_lig,
        sanitize           = sanitize,
        largest_frag       = largest_frag,
        relax_iter         = relax_iter,
        num_sampling_steps = num_sampling_steps,
        resamplings        = inpaint_resamplings,
        jump_length        = jump_length,            # ← NEW
        repulsion_scale    = repulsion_scale,         # ← NEW
        repulsion_cutoff   = repulsion_cutoff,        # ← NEW
        ligand_inp         = ligand_inp,
        lig_mask_fixed     = lig_mask_fixed,
        **karras_kwargs,
    )

    return repaired_mols


# ═══════════════════════════════════════════════════════════════════════════════
# Pocket preparation helper (for Phase-2 size sampling)
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_pocket(model, pdb_file, ref_ligand, pocket_ids):
    """
    Parse the PDB, identify the pocket residues, and prepare a single-sample
    pocket dict so we can read ``pocket['size']`` for the size distribution.
    """
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]

    if pocket_ids is not None:
        residues = [
            pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
            for x in pocket_ids
        ]
    else:
        residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

    pocket_single = model.prepare_pocket(residues, repeats=1)
    return pocket_single


def _postprocess_molecule(
    mol: Chem.Mol,
    *,
    n_fixed: int,
    is_inpainting: bool,
    sanitize: bool,
    relax_iter: int,
    largest_frag: bool,
) -> Optional[Chem.Mol]:
    """
    Post-process a single molecule.

    When ``is_inpainting=True`` and constrained_relax is available,
    uses scaffold-pinned MMFF relaxation (only generated atoms move).
    Otherwise falls back to standard process_molecule.
    """
    if mol is None:
        return None

    if is_inpainting and n_fixed > 0 and _HAS_CONSTRAINED_RELAX:
        try:
            return process_molecule_inpaint(
                mol,
                n_fixed=n_fixed,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag,
            )
        except Exception:
            return None
    else:
        try:
            return process_molecule(
                mol,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag,
            )
        except Exception:
            return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Two-phase de novo / inpainting ligand generation.\n\n"
            "Phase 1 : generates molecules from pure noise (or inpaints with\n"
            "          --fix_atoms).\n"
            "Phase 2 : automatic fragment growing via conditional inpainting\n"
            "          (enabled with --refine_fragments). Multi-fragment\n"
            "          molecules are trimmed to their largest fragment, then\n"
            "          grown if below --min_heavy_atoms. Uses the model's\n"
            "          learned p(N_lig | N_pocket) distribution (DiffSBDD\n"
            "          approach) to decide how many atoms to add."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required paths ────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the LigandPocketEDM checkpoint (.ckpt).")
    parser.add_argument("--pdbfile",    type=str,  required=True,
                        help="Path to the target protein PDB file.")
    parser.add_argument("--ref_ligand", type=str,  default=None,
                        help=(
                            "SDF file path  →  used to define the pocket.\n"
                            "'<chain>:<resi>'  →  pocket residue in the PDB "
                            "(required when --fix_atoms contains atom names)."
                        ))
    parser.add_argument("--pocket_ids", nargs="+", default=None,
                        help=(
                            "Explicit pocket residues as '<chain>:<resi>' pairs "
                            "(e.g.  A:100 A:101 A:150).  When provided, the "
                            "pocket is built from these residues directly, "
                            "bypassing the ref_ligand-based distance cutoff.\n"
                            "Useful with --fix_atoms <SDF> when the fragment "
                            "SDF is not co-located with the original ref_ligand."
                        ))
    parser.add_argument("--outfile",    type=Path, required=True,
                        help="Output SDF file path for the generated molecules.")

    # ── Phase-1 manual inpainting (optional) ─────────────────────────────────
    parser.add_argument("--fix_atoms", nargs="+", default=None,
                        help=(
                            "Atoms to hold fixed during Repaint inpainting.\n"
                            "Pass a single SDF path   → fix all atoms in that file.\n"
                            "Pass atom names (C1 N2 …) → fix those atoms from the "
                            "--ref_ligand PDB residue."
                        ))
    parser.add_argument("--add_n_nodes", type=int, default=None,
                    help=(
                        "Number of extra atoms the model should generate "
                        "on top of the fixed substructure in Phase-1 "
                        "inpainting. If omitted, the model's learned "
                        "p(N_lig | N_pocket) distribution is used to "
                        "sample a suitable total size."
                    ))

    # ── Phase-2: Conditional Inpainting Pipeline ─────────────────────────────
    parser.add_argument("--refine_fragments", action="store_true",
                        help=(
                            "Enable Phase-2 conditional inpainting to grow "
                            "small molecules from Phase-1 output. Multi-fragment "
                            "molecules are first trimmed to their largest "
                            "fragment, then grown if below --min_heavy_atoms."
                        ))
    parser.add_argument("--min_heavy_atoms", type=int, default=15,
                        help=(
                            "Minimum heavy-atom count for a single-fragment "
                            "molecule to pass (Group C). Smaller molecules are "
                            "routed to fragment growing (Group B). (default: 15)"
                        ))
    parser.add_argument("--inpaint_resamplings", type=int, default=5,
                        help=(
                            "Repaint r for the Phase-2 inpainting pass. "
                            "Higher values improve connectivity quality at "
                            "the cost of speed. (default: 5)"
                        ))
    parser.add_argument("--p2_max_rounds", type=int, default=3,
                        help=(
                            "Maximum number of iterative refinement rounds "
                            "in Phase 2. After each round, molecules are "
                            "re-assessed: connected ones graduate, "
                            "disconnected ones are re-queued. (default: 3)"
                        ))

    # ── Sampling configuration ────────────────────────────────────────────────
    parser.add_argument("--n_samples",          type=int, default=100,
                        help="Target number of valid molecules to collect (default: 100).")
    parser.add_argument("--batch_size",         type=int, default=60,
                        help="Molecules requested per Phase-1 batch (default: 60).")
    parser.add_argument("--num_sampling_steps", type=int, default=40,
                        help="Karras denoising steps (default: 40).")
    parser.add_argument("--n_nodes_bias",       type=int, default=0,
                        help="The number of atoms adding in sampling distribution.")
    parser.add_argument("--n_nodes_min",        type=int, default=0,
                        help="The number of minimum atom.")
    parser.add_argument("--resamplings",        type=int, default=10,
                        help="Repaint r for Phase-1 (default: 10).")
    parser.add_argument("--sanitize",           action="store_true",
                        help="Apply RDKit sanitization in process_molecule.")
    parser.add_argument("--relax",              action="store_true",
                        help="Apply 200 MMFF relaxation steps in process_molecule.")
    parser.add_argument("--largest_frag",       action="store_true",
                        help="Keep only the largest fragment from each molecule.")

    parser.add_argument("--jump_length", type=int, default=1,
                        help=(
                            "Repaint j — chunk size for backward jumps in the "
                            "denoising schedule.  j=1 means no jumps (original "
                            "behaviour).  Paper recommends j=10 with r=10.  "
                            "Higher values improve boundary harmonisation at "
                            "the cost of extra compute.  (default: 1)"
                        ))
    parser.add_argument("--repulsion_scale", type=float, default=0.0,
                        help=(
                            "Steric repulsion guidance strength applied during "
                            "the last portion of denoising to push generated "
                            "atoms away from the fixed scaffold.  0 = off.  "
                            "Typical range: 0.005–0.02.  (default: 0.0)"
                        ))
    parser.add_argument("--repulsion_cutoff", type=float, default=1.2,
                        help=(
                            "Distance cutoff (in normalised space ≈ Angstrom) "
                            "below which repulsion is applied.  (default: 1.2)"
                        ))

    # ── Karras noise-schedule overrides (all optional) ────────────────────────
    parser.add_argument("--sigma_min_pos",  type=float, default=None)
    parser.add_argument("--sigma_max_pos",  type=float, default=None)
    parser.add_argument("--sigma_min_feat", type=float, default=None)
    parser.add_argument("--sigma_max_feat", type=float, default=None)
    parser.add_argument("--rho",            type=float, default=None)
    parser.add_argument("--tau_start",      type=float, default=None)
    parser.add_argument("--tau_end",        type=float, default=None)

    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(args.pdbfile).is_file():
        raise FileNotFoundError(f"PDB file not found: {args.pdbfile}")
    if not Path(args.ref_ligand).is_file() and ":" not in args.ref_ligand:
        raise FileNotFoundError(
            f"--ref_ligand must be an existing SDF path or '<chain>:<resi>', "
            f"got: {args.ref_ligand}"
        )
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    use_inpainting = args.fix_atoms is not None

    if use_inpainting and not args.fix_atoms[0].endswith(".sdf"):
        if ":" not in args.ref_ligand:
            raise ValueError(
                "When --fix_atoms contains atom names (not an SDF), "
                "--ref_ligand must be in '<chain>:<resi>' format."
            )

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init]  device             : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[init]  checkpoint         : {args.checkpoint}")
    model = LigandPocketEDM.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model = model.to(device)
    model.eval()
    print(f"[init]  model ready")

    # ── Phase-1 inpainting setup (optional) ───────────────────────────────────
    fixed_coord   = None
    fixed_one_hot = None

    if use_inpainting:
        pdb_model = PDBParser(QUIET=True).get_structure("", args.pdbfile)[0]
        fixed_coord, fixed_one_hot = prepare_substructure(
            args.ref_ligand, args.fix_atoms, pdb_model, model.lig_type_encoder,
        )
        n_fixed = fixed_coord.shape[0]
        print(
            f"[init]  inpainting mode    : {n_fixed} fixed atom(s) from "
            f"{'SDF' if args.fix_atoms[0].endswith('.sdf') else 'PDB'}"
        )
        if args.jump_length > 1:
            print(f"[init]  jump_length        : {args.jump_length}")
        if args.repulsion_scale > 0:
            print(f"[init]  repulsion_scale    : {args.repulsion_scale}")
            print(f"[init]  repulsion_cutoff   : {args.repulsion_cutoff}")
        if _HAS_CONSTRAINED_RELAX and args.relax:
            print(f"[init]  constrained relax  : ON (scaffold pinned)")
        pocket_ref_ligand = args.ref_ligand if Path(args.ref_ligand).is_file() else None
        pocket_ids        = [args.ref_ligand] if ":" in args.ref_ligand else None
    else:
        pocket_ref_ligand = args.ref_ligand
        pocket_ids        = None
        n_fixed           = 0
        print(f"[init]  mode               : de novo")

    if args.refine_fragments:
        print(f"[init]  Phase-2 refinement : ON")
        print(f"         min_heavy_atoms   : {args.min_heavy_atoms}")
        print(f"         inpaint_resamplings: {args.inpaint_resamplings}")
        print(f"         p2_max_rounds     : {args.p2_max_rounds}")

    print()

    # ── Shared settings ───────────────────────────────────────────────────────
    relax_iter = 200 if args.relax else 0

    karras_kwargs = dict(
        sigma_min_pos  = args.sigma_min_pos,
        sigma_max_pos  = args.sigma_max_pos,
        sigma_min_feat = args.sigma_min_feat,
        sigma_max_feat = args.sigma_max_feat,
        rho            = args.rho,
        tau_start      = args.tau_start,
        tau_end        = args.tau_end,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 – De-novo / manual-inpainting generation
    # ══════════════════════════════════════════════════════════════════════════
    p1_all_valid: List[Chem.Mol] = []
    n_generated_p1 = 0
    n_valid_p1     = 0

    mode_tag = "inpaint" if use_inpainting else "de novo"
    pbar = tqdm(total=args.n_samples, desc=f"Phase 1 [{mode_tag}]", unit="mol")

    while len(p1_all_valid) < args.n_samples:
        try:
            if use_inpainting:
                batch   = args.batch_size

                if args.add_n_nodes is not None:
                    add_n = args.add_n_nodes
                else:
                    if not hasattr(args, '_cached_pocket_size'):
                        _pocket_tmp = resolve_pocket(
                            model, args.pdbfile, args.ref_ligand, pocket_ids
                        )
                        args._cached_pocket_size = _pocket_tmp['size']

                    pocket_sz = args._cached_pocket_size
                    sampled_n = model.model.size_distribution.sample_conditional(
                        n1=None,
                        n2=pocket_sz.expand(batch),
                    )                             # shape [batch]
                    sampled_n = torch.clamp(sampled_n, min=n_fixed + 1)
                    add_n_per_sample = (sampled_n - n_fixed).tolist()

                n_types = fixed_one_hot.shape[-1]
                all_x       = []
                all_one_hot = []
                all_mask    = []
                all_fixed   = []
                sizes_list  = []

                for i in range(batch):
                    an = add_n if args.add_n_nodes is not None else add_n_per_sample[i]
                    n_total = n_fixed + an

                    x_pad = torch.zeros(n_total, 3, dtype=FLOAT_TYPE, device=device)
                    x_pad[:n_fixed] = fixed_coord.to(device)

                    h_pad = torch.zeros(n_total, n_types, dtype=FLOAT_TYPE, device=device)
                    h_pad[:n_fixed] = fixed_one_hot.to(FLOAT_TYPE).to(device)

                    mask_i = torch.full((n_total,), i, dtype=INT_TYPE, device=device)

                    fixed_i = torch.zeros(n_total, dtype=FLOAT_TYPE, device=device)
                    fixed_i[:n_fixed] = 1.0

                    all_x.append(x_pad)
                    all_one_hot.append(h_pad)
                    all_mask.append(mask_i)
                    all_fixed.append(fixed_i)
                    sizes_list.append(n_total)

                ligand_inp_p1 = {
                    "x":       torch.cat(all_x,       dim=0),
                    "one_hot": torch.cat(all_one_hot,  dim=0),
                    "size":    torch.tensor(sizes_list, dtype=INT_TYPE, device=device),
                    "mask":    torch.cat(all_mask,      dim=0),
                }
                lig_mask_fixed_p1 = torch.cat(all_fixed, dim=0)

                p1_raw = model.generate_ligands(
                    pdb_file           = args.pdbfile,
                    n_samples          = batch,
                    ref_ligand         = pocket_ref_ligand,
                    pocket_ids         = pocket_ids,
                    num_nodes_lig      = ligand_inp_p1["size"],
                    sanitize           = False,
                    largest_frag       = False,
                    relax_iter         = 0,
                    num_sampling_steps = args.num_sampling_steps,
                    n_nodes_bias       = 0,
                    n_nodes_min        = 0,
                    resamplings        = args.resamplings,
                    jump_length        = args.jump_length,        # ← NEW
                    repulsion_scale    = args.repulsion_scale,    # ← NEW
                    repulsion_cutoff   = args.repulsion_cutoff,   # ← NEW
                    ligand_inp         = ligand_inp_p1,
                    lig_mask_fixed     = lig_mask_fixed_p1,
                    **karras_kwargs,
                )
            else:
                p1_raw = model.generate_ligands(
                    pdb_file           = args.pdbfile,
                    n_samples          = args.batch_size,
                    ref_ligand         = pocket_ref_ligand,
                    pocket_ids         = pocket_ids,
                    sanitize           = False,
                    largest_frag       = False,
                    relax_iter         = 0,
                    num_sampling_steps = args.num_sampling_steps,
                    n_nodes_bias       = args.n_nodes_bias,
                    n_nodes_min        = args.n_nodes_min,
                    resamplings        = args.resamplings,
                    **karras_kwargs,
                )

        except Exception as exc:
            print(f"\n[warn]  Phase-1 batch failed: {exc} — retrying.")
            continue

        n_generated_p1 += args.batch_size

        p1_valid_batch: List[Chem.Mol] = []
        for mol in p1_raw:
            processed = _postprocess_molecule(
                mol,
                n_fixed=n_fixed,
                is_inpainting=use_inpainting,
                sanitize=args.sanitize,
                relax_iter=relax_iter,
                largest_frag=(False if args.refine_fragments
                              else args.largest_frag),
            )
            if processed is not None:
                p1_valid_batch.append(processed)

        n_valid_p1 += len(p1_valid_batch)

        remaining = args.n_samples - len(p1_all_valid)
        take      = p1_valid_batch[:remaining]
        p1_all_valid.extend(take)
        pbar.update(len(take))

        p1_rate = n_valid_p1 / n_generated_p1 * 100 if n_generated_p1 else 0.0
        pbar.set_postfix(dict(
            generated = n_generated_p1,
            P1_pass   = f"{p1_rate:.1f}%",
            collected = len(p1_all_valid),
        ))

    pbar.close()

    if args.refine_fragments:
        print(f"\n[Phase 2]  Assessing {len(p1_all_valid)} Phase-1 molecules …")

        # ── §1  Initial Assessment & Routing ──────────────────────────────
        group_b, group_c, n_trimmed = assess_and_route_molecules(
            p1_all_valid,
            min_heavy_atoms = args.min_heavy_atoms,
            sanitize        = args.sanitize,
            n_fixed         = n_fixed,
        )
        n_b, n_c = len(group_b), len(group_c)
        print(f"[Phase 2]  Initial routing (no-link strategy):")
        print(f"           Trimmed (kept largest frag) : {n_trimmed}")
        print(f"           Group B (grow)              : {n_b}")
        print(f"           Group C (pass)              : {n_c}")

        # Prepare pocket once (shared across all rounds)
        pocket_single = None
        pocket_size   = None
        if group_b:
            pocket_single = resolve_pocket(
                model, args.pdbfile, args.ref_ligand, pocket_ids
            )
            pocket_size = pocket_single['size']
            print(f"[Phase 2]  Pocket size: {pocket_size.item()}")

        # Accumulators
        graduated_mols: List[Chem.Mol] = list(group_c)  # Group C passes directly
        failed_mols:    List[Chem.Mol] = []              # gave up after max_rounds

        # Queue = molecules still needing growing
        repair_queue: List[dict] = list(group_b)

        # ── §2  Iterative refinement loop ─────────────────────────────────
        max_rounds = args.p2_max_rounds

        for round_idx in range(1, max_rounds + 1):
            if not repair_queue:
                break

            n_in_queue = len(repair_queue)
            print(f"\n[Phase 2]  ── Round {round_idx}/{max_rounds} "
                  f"({n_in_queue} molecule(s) in queue) ──")

            # §2a – Sample target sizes for this round
            add_n_list = sample_target_sizes(
                model, pocket_size, repair_queue
            )

            for i, (entry, add_n) in enumerate(zip(repair_queue, add_n_list)):
                print(
                    f"           [grow] mol {i}: "
                    f"N_fixed={entry['heavy_atoms']}, "
                    f"add_n={add_n}, "
                    f"N_total={entry['heavy_atoms'] + add_n}"
                )

            # §2b – Run inpainting in batches
            round_output: List[Chem.Mol] = []
            p2_batch_size = args.batch_size

            try:
                for i in range(0, len(repair_queue), p2_batch_size):
                    chunk_entries = repair_queue[i : i + p2_batch_size]
                    chunk_add_n  = add_n_list[i : i + p2_batch_size]

                    print(f"           Batch {i // p2_batch_size + 1} "
                          f"({len(chunk_entries)} mols) ...")

                    chunk_mols = run_refinement(
                        model               = model,
                        entries             = chunk_entries,
                        add_n_list          = chunk_add_n,
                        pdb_file            = args.pdbfile,
                        ref_ligand          = args.ref_ligand,
                        pocket_ids          = pocket_ids,
                        device              = device,
                        inpaint_resamplings = args.inpaint_resamplings,
                        jump_length         = args.jump_length,        # ← NEW
                        repulsion_scale     = args.repulsion_scale,    # ← NEW
                        repulsion_cutoff    = args.repulsion_cutoff,   # ← NEW
                        num_sampling_steps  = args.num_sampling_steps,
                        sanitize            = args.sanitize,
                        relax_iter          = relax_iter,
                        largest_frag        = False,  # keep all frags for re-assessment
                        karras_kwargs       = karras_kwargs,
                    )
                    round_output.extend(chunk_mols)

            except Exception as exc:
                print(f"[Phase 2]  Round {round_idx} inpainting failed: {exc}")
                round_output = []

            # §2c – Re-assess: which molecules are now large enough?
            still_small, now_good, n_trimmed_round = assess_and_route_molecules(
                round_output,
                min_heavy_atoms = args.min_heavy_atoms,
                sanitize        = args.sanitize,
                n_fixed         = n_fixed,
            )

            n_good    = len(now_good)
            n_retry   = len(still_small)
            n_lost    = n_in_queue - len(round_output)

            print(f"[Phase 2]  Round {round_idx} results:")
            print(f"           Graduated (pass)        : {n_good}")
            print(f"           Still too small (retry)  : {n_retry}")
            if n_trimmed_round > 0:
                print(f"           Trimmed (kept largest)   : {n_trimmed_round}")
            if n_lost > 0:
                print(f"           Lost (invalid/failed)    : {n_lost}")

            # Collect graduated molecules
            graduated_mols.extend(now_good)

            # Re-queue small ones for next round
            repair_queue = still_small

            if not repair_queue:
                print(f"[Phase 2]  All molecules graduated after round {round_idx}!")
                break

        # ── §3  Handle molecules that never grew large enough ──────────
        if repair_queue:
            print(f"\n[Phase 2]  {len(repair_queue)} molecule(s) still too "
                  f"small after {max_rounds} round(s).")

            # Keep them as best-effort output (already single-fragment)
            for entry in repair_queue:
                mol = entry.get('mol')
                if mol is not None:
                    failed_mols.append(mol)

            print(f"           Kept {len(failed_mols)} as best-effort output")

        # ── Merge all results ─────────────────────────────────────────────
        valid_molecules = graduated_mols + failed_mols

        n_repaired = len(graduated_mols) - n_c  # exclude original Group C
        print(f"\n[Phase 2]  Final summary:")
        print(f"           Trimmed (largest frag)    : {n_trimmed}")
        print(f"           Graduated (grown → pass)  : {n_repaired} / {n_b}")
        print(f"           Passed (Group C)           : {n_c}")
        print(f"           Best-effort (still small)  : {len(failed_mols)}")
        print(f"           Total output               : {len(valid_molecules)}")

    else:
        valid_molecules = p1_all_valid

    # ── Universal final filter: guarantee single-fragment output ──────
    if args.largest_frag or use_inpainting:
        final_mols = []
        for mol in valid_molecules:
            try:
                frag_indicies = Chem.GetMolFrags(mol, asMols=False)
                if len(frag_indicies) > 1:
                    if use_inpainting:
                        frag_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                        for f_mol, f_idx in zip(frag_mols, frag_indicies):
                            if 0 in f_idx:
                                mol = f_mol
                                break
                        if args.sanitize:
                            Chem.SanitizeMol(mol)
                    else:
                        if args.largest_frag:
                            mol = process_molecule(
                                mol, sanitize=args.sanitize,
                                relax_iter=0, largest_frag=True,
                            )
                if mol is not None:
                    final_mols.append(mol)
            except Exception:
                final_mols.append(mol)
        n_final_dropped = len(valid_molecules) - len(final_mols)
        if n_final_dropped > 0:
            print(f"[final]  largest_frag safety net removed "
                  f"{n_final_dropped} disconnected molecule(s)")
        valid_molecules = final_mols

    utils.write_sdf_file(args.outfile, valid_molecules)

    p1_rate = n_valid_p1 / n_generated_p1 * 100 if n_generated_p1 else 0.0

    fix_line = ""
    if use_inpainting:
        src = "SDF" if args.fix_atoms[0].endswith(".sdf") else "PDB atom names"
        fix_line = (
            f"\n  Fixed atoms        : {fixed_coord.shape[0]} "
            f"(from {src}: {args.fix_atoms})"
            f"\n  resamplings / J    : {args.resamplings} / {args.jump_length}"
        )
        if args.repulsion_scale > 0:
            fix_line += f"\n  repulsion          : scale={args.repulsion_scale}, cutoff={args.repulsion_cutoff}"

    refine_line = ""
    if args.refine_fragments:
        refine_line = (
            f"\n  ── Phase-2 refinement (grow-only) ──"
            f"\n  Trimmed (largest frag): {n_trimmed}"
            f"\n  Group B (grow)     : {n_b}"
            f"\n  Group C (pass)     : {n_c}"
            f"\n  Graduated          : {n_repaired} / {n_b}"
            f"\n  Best-effort        : {len(failed_mols)}"
            f"\n  Max rounds         : {args.p2_max_rounds}"
            f"\n  Inpaint resamplings: {args.inpaint_resamplings}"
            f"\n  Size sampling      : pocket-conditioned (DiffSBDD)"
        )

    print(
        f"\n{'=' * 62}\n"
        f"  Target             : {Path(args.pdbfile).name}\n"
        f"  Pocket / ref       : {args.ref_ligand}\n"
        f"  Mode               : {mode_tag}"
        f"{fix_line}\n"
        f"  Total generated P1 : {n_generated_p1}\n"
        f"  Phase-1 pass rate  : {p1_rate:.1f}%"
        f"  ({n_valid_p1} / {n_generated_p1})"
        f"{refine_line}\n"
        f"  Molecules saved    : {len(valid_molecules)}\n"
        f"  Output SDF         : {args.outfile}\n"
        f"{'=' * 62}"
    )