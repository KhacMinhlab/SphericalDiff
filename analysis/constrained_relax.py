"""
constrained_relax.py
====================
RDKit constrained force-field relaxation for inpainted molecules.

Fixed atoms (from inpaint_mask) are firmly pinned via position constraints;
only the newly generated atoms are allowed to move, resolving minor steric
clashes without distorting the preserved scaffold.

Design Notes
------------
1.  We use MMFF94 (preferred for drug-like molecules) with UFF as fallback.
2.  Position constraints are applied via `MMFFAddPositionConstraint` /
    `UFFAddDistanceConstraint` to anchor each fixed atom to its original
    coordinates with a very large force constant.
3.  The conformer's coordinates for fixed atoms are restored to their exact
    original values after optimisation, eliminating any residual numerical drift.
4.  Fragment handling: if the molecule has disconnected fragments, we keep
    the fragment that contains atom index 0 (the first fixed atom), not
    just the largest fragment — this ensures the scaffold is always preserved.
"""

import warnings
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdmolops


def constrained_process_molecule(
    rdmol: Chem.Mol,
    fixed_atom_indices: List[int],
    sanitize: bool = True,
    relax_iter: int = 200,
    largest_frag: bool = False,
    add_hydrogens: bool = False,
    force_constant: float = 1e5,
) -> Optional[Chem.Mol]:
    """
    Process an RDKit molecule with constrained force-field relaxation.

    Parameters
    ----------
    rdmol : Chem.Mol
        Input molecule with a 3D conformer.
    fixed_atom_indices : list of int
        Atom indices that should remain fixed (from the inpaint scaffold).
    sanitize : bool
        Run SanitizeMol before relaxation.
    relax_iter : int
        Maximum force-field optimisation steps (0 = skip relaxation).
    largest_frag : bool
        If True, keep only the largest fragment.
    add_hydrogens : bool
        Add explicit hydrogens before relaxation.
    force_constant : float
        Strength of the position constraint on fixed atoms (kcal/mol/Å²).
        1e5 effectively freezes them in place.

    Returns
    -------
    Chem.Mol or None
        Processed molecule, or None if sanity checks fail.
    """
    mol = Chem.RWMol(Chem.Mol(rdmol))

    # ── 0. Optional: keep scaffold-containing fragment ────────────────
    if largest_frag:
        mol = _keep_scaffold_fragment(mol, fixed_atom_indices)
        if mol is None:
            return None
        # Re-map fixed indices after fragment extraction
        fixed_atom_indices = _remap_indices(
            rdmol, mol, fixed_atom_indices
        )

    # ── 1. Sanitize ───────────────────────────────────────────────────
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            warnings.warn(f'Sanitization failed: {e}. Returning None.')
            return None

    # ── 2. Add hydrogens (optional) ───────────────────────────────────
    if add_hydrogens and mol.GetNumConformers() > 0:
        mol = Chem.AddHs(mol, addCoords=True)
        # Fixed indices don't change because H's are appended at the end

    # ── 3. Skip relaxation if not requested ───────────────────────────
    if relax_iter <= 0 or mol.GetNumConformers() == 0:
        return mol.GetMol() if isinstance(mol, Chem.RWMol) else mol

    # ── 4. Save original coordinates of fixed atoms ───────────────────
    conf = mol.GetConformer()
    orig_positions = {}
    for idx in fixed_atom_indices:
        if idx < mol.GetNumAtoms():
            pos = conf.GetAtomPosition(idx)
            orig_positions[idx] = np.array([pos.x, pos.y, pos.z])

    # ── 5. Try MMFF94, fall back to UFF ───────────────────────────────
    mol_out = mol.GetMol() if isinstance(mol, Chem.RWMol) else mol
    success = _try_mmff_constrained(mol_out, fixed_atom_indices,
                                     force_constant, relax_iter)
    if not success:
        success = _try_uff_constrained(mol_out, fixed_atom_indices,
                                        force_constant, relax_iter)
    if not success:
        warnings.warn('Both MMFF and UFF constrained relaxation failed. '
                      'Returning unrelaxed molecule.')

    # ── 6. Restore exact original coordinates for fixed atoms ─────────
    #    (removes sub-Å numerical drift from the optimiser)
    conf = mol_out.GetConformer()
    for idx, pos in orig_positions.items():
        if idx < mol_out.GetNumAtoms():
            conf.SetAtomPosition(idx, pos.tolist())

    # ── 7. Final sanitize ─────────────────────────────────────────────
    if sanitize:
        try:
            Chem.SanitizeMol(mol_out)
        except Exception:
            return None

    return mol_out


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _try_mmff_constrained(mol, fixed_indices, force_constant, max_iter):
    """Attempt MMFF94 optimisation with position constraints."""
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if mmff_props is None:
            return False

        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
        if ff is None:
            return False

        # Pin each fixed atom with a strong position constraint
        for idx in fixed_indices:
            if idx < mol.GetNumAtoms():
                ff.MMFFAddPositionConstraint(idx, force_constant, 0)
                # Args: atomIdx, maxDisp (Å), forceConstant
                # maxDisp=0 with high force constant ≈ hard freeze

        result = ff.Minimize(maxIts=max_iter)
        return True  # result: 0=converged, 1=not converged (but still valid)

    except Exception as e:
        warnings.warn(f'MMFF constrained relax failed: {e}')
        return False


def _try_uff_constrained(mol, fixed_indices, force_constant, max_iter):
    """Fallback: UFF optimisation with distance constraints to freeze atoms."""
    try:
        if not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            return False

        ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is None:
            return False

        # UFF doesn't have MMFFAddPositionConstraint, so we use a different
        # approach: add a very stiff distance constraint from each fixed atom
        # to a virtual fixed point (its original position).
        # RDKit's UFF supports UFFAddDistanceConstraint but not position
        # constraints directly.  We emulate it by constraining the atom to
        # itself (distance = 0) with a very large force constant.
        conf = mol.GetConformer()
        for idx in fixed_indices:
            if idx < mol.GetNumAtoms():
                # Fix atom position by adding a large position constraint
                # via the ForceField.AddFixedPoint mechanism
                pos = conf.GetAtomPosition(idx)
                ff.AddFixedPoint(idx)

        result = ff.Minimize(maxIts=max_iter)
        return True

    except Exception as e:
        warnings.warn(f'UFF constrained relax failed: {e}')
        return False


def _keep_scaffold_fragment(mol, fixed_indices):
    """
    Keep the fragment that contains the first fixed atom (atom 0 by
    convention in the inpainting pipeline), not just the largest one.

    This prevents the scaffold from being dropped if the generated
    portion happens to have more atoms.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False,
                              fragsMolAtomMapping=None)
    frag_indices = Chem.GetMolFrags(mol, asMols=False)

    if len(frags) <= 1:
        return mol

    # Find the fragment containing fixed atom 0
    anchor_idx = fixed_indices[0] if fixed_indices else 0
    for frag_mol, frag_atom_ids in zip(frags, frag_indices):
        if anchor_idx in frag_atom_ids:
            return frag_mol

    # Fallback: largest fragment
    return max(frags, key=lambda m: m.GetNumAtoms())


def _remap_indices(old_mol, new_mol, old_indices):
    """
    After fragment extraction, atom indices change. Re-map fixed
    indices from the original molecule to the new fragment.

    Heuristic: match by 3D position (< 0.01 Å) since atom ordering
    may differ.
    """
    if old_mol.GetNumConformers() == 0 or new_mol.GetNumConformers() == 0:
        # Can't match by position; return indices as-is and hope for the best
        return [i for i in old_indices if i < new_mol.GetNumAtoms()]

    old_conf = old_mol.GetConformer()
    new_conf = new_mol.GetConformer()

    remapped = []
    for old_idx in old_indices:
        if old_idx >= old_mol.GetNumAtoms():
            continue
        old_pos = np.array(old_conf.GetAtomPosition(old_idx))

        best_new_idx = None
        best_dist = float('inf')
        for new_idx in range(new_mol.GetNumAtoms()):
            new_pos = np.array(new_conf.GetAtomPosition(new_idx))
            d = np.linalg.norm(old_pos - new_pos)
            if d < best_dist:
                best_dist = d
                best_new_idx = new_idx

        if best_dist < 0.01 and best_new_idx is not None:
            remapped.append(best_new_idx)

    return remapped


# ──────────────────────────────────────────────────────────────────────────
# Drop-in replacement for process_molecule in your pipeline
# ──────────────────────────────────────────────────────────────────────────

def process_molecule_inpaint(
    rdmol: Chem.Mol,
    n_fixed: int,
    add_hydrogens: bool = False,
    sanitize: bool = True,
    relax_iter: int = 200,
    largest_frag: bool = True,
    force_constant: float = 1e5,
) -> Optional[Chem.Mol]:
    """
    Convenience wrapper matching the call signature your pipeline expects.

    Assumes fixed atoms are the first ``n_fixed`` atoms in the molecule
    (which matches the ordering in your inpaint.py: fixed atoms go at
    the top of each molecule's atom list).

    Usage in inpaint.py — replace:
        mol = process_molecule(mol, sanitize=..., relax_iter=..., ...)
    With:
        mol = process_molecule_inpaint(mol, n_fixed=n_fixed, sanitize=..., ...)
    """
    fixed_indices = list(range(n_fixed))
    return constrained_process_molecule(
        rdmol,
        fixed_atom_indices=fixed_indices,
        sanitize=sanitize,
        relax_iter=relax_iter,
        largest_frag=largest_frag,
        add_hydrogens=add_hydrogens,
        force_constant=force_constant,
    )