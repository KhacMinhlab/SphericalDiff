import argparse
import glob
import os
import sys
from pathlib import Path
 
import pandas as pd
 
try:
    from posebusters import PoseBusters
except ImportError:
    print("ERROR: posebusters not installed. Run: pip install posebusters")
    sys.exit(1)
 
 
def match_sdf_to_pdb(sdf_path: str, protein_test_dir: str, dataset: str) -> str:
    """
    Match an SDF file to its corresponding protein PDB file
    using the same logic as GCDM.
    """
    stem = Path(sdf_path).stem
 
    if dataset == "bindingmoad":
        pattern = stem.split("_")[0] + "*.pdb"
    elif dataset == "crossdocked":
        pattern = "-".join(stem.split("-")[:2]) + "*.pdb"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
 
    matches = glob.glob(os.path.join(protein_test_dir, pattern))
 
    if not matches:
        return None
 
    return matches[0]
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Run PoseBusters evaluation on generated molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_molecule_dir", type=str, required=True,
        help="Directory containing generated .sdf files (e.g. DiffSBDD processed/ folder)"
    )
    parser.add_argument(
        "--input_protein_dir", type=str, required=True,
        help=(
            "Root protein directory. PDB files must be in {input_protein_dir}/test/\n"
            "For crossdocked: typically .../processed_crossdock_noH_full_temp\n"
            "For bindingmoad: typically .../processed_noH_full"
        )
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["crossdocked", "bindingmoad"],
        help="Dataset name (determines SDF->PDB matching logic)"
    )
    parser.add_argument(
        "--outdir", type=str, default="./posebusters_output",
        help="Output directory for CSV results (default: ./posebusters_output)"
    )
    parser.add_argument(
        "--full_report", action="store_true", default=True,
        help="Generate full PoseBusters report with all individual tests (default: True)"
    )
    parser.add_argument(
        "--config", type=str, default="dock",
        choices=["dock", "mol"],
        help=(
            "PoseBusters config mode (default: dock).\n"
            "  dock = includes protein-ligand clash checks (requires PDB)\n"
            "  mol  = molecule-only checks (no protein needed)"
        )
    )
    args = parser.parse_args()
 
    # ── Validate paths ──
    assert os.path.isdir(args.input_molecule_dir), \
        f"Molecule directory not found: {args.input_molecule_dir}"
    
    protein_test_dir = os.path.join(args.input_protein_dir, "test")
    assert os.path.isdir(protein_test_dir), \
        f"Protein test directory not found: {protein_test_dir}\n" \
        f"PDB files must be in {{input_protein_dir}}/test/"
 
    # ── Create output directory ──
    os.makedirs(args.outdir, exist_ok=True)
 
    # ── Find SDF files ──
    sdf_files = sorted(glob.glob(os.path.join(args.input_molecule_dir, "*.sdf")))
    assert sdf_files, f"No .sdf files found in {args.input_molecule_dir}"
    print(f"Found {len(sdf_files)} SDF files in {args.input_molecule_dir}")
 
    # ── Match SDF -> PDB ──
    mol_pred_list = []
    mol_cond_list = []
    skipped = []
 
    for sdf_path in sdf_files:
        pdb_path = match_sdf_to_pdb(sdf_path, protein_test_dir, args.dataset)
        if pdb_path is None:
            skipped.append(Path(sdf_path).stem)
            continue
        mol_pred_list.append(sdf_path)
        mol_cond_list.append(pdb_path)
 
    if skipped:
        print(f"WARNING: Skipped {len(skipped)} SDF files (no matching PDB):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
 
    assert mol_pred_list, "No SDF-PDB pairs matched. Check naming convention."
    print(f"Matched {len(mol_pred_list)} SDF-PDB pairs")
 
    # ── Build molecule table ──
    mol_table = pd.DataFrame({
        "mol_pred": mol_pred_list,
        "mol_true": None,
        "mol_cond": mol_cond_list,
    })
 
    # ── Run PoseBusters ──
    print(f"\nRunning PoseBusters (config={args.config}, full_report={args.full_report})...")
    print("This may take a while...\n")
 
    buster = PoseBusters(config=args.config, top_n=None)
    bust_results = buster.bust_table(mol_table, full_report=args.full_report)
 
    # ── Save full results ──
    csv_path = os.path.join(args.outdir, "posebusters_full_results.csv")
    bust_results.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")
 
    # ── Print summary ──
    bool_cols = bust_results.select_dtypes(include="bool").columns.tolist()
 
    if not bool_cols:
        print("WARNING: No boolean columns found. Check PoseBusters output.")
        return
 
    print("\n" + "=" * 70)
    print(f"  PoseBusters Summary — {args.dataset}")
    print(f"  Config: {args.config} | Molecules evaluated: {len(bust_results)}")
    print("=" * 70)
 
    summary_rows = []
    for col in bool_cols:
        n_pass = bust_results[col].sum()
        n_total = len(bust_results)
        rate = n_pass / n_total * 100
        print(f"  {col:45s}  {n_pass:5d}/{n_total:5d}  ({rate:5.1f}%)")
        summary_rows.append({"test": col, "n_pass": n_pass, "n_total": n_total, "pass_rate": rate})
 
    # ── PB-Valid: pass ALL ──
    all_pass = bust_results[bool_cols].all(axis=1)
    n_all = all_pass.sum()
    rate_all = n_all / len(bust_results) * 100
    print("-" * 70)
    print(f"  {'PB-Valid (pass ALL tests)':45s}  {n_all:5d}/{len(bust_results):5d}  ({rate_all:5.1f}%)")
 
    # ── PB-Valid without protein-ligand steric clash ──
    prot_clash_cols = [c for c in bool_cols if "protein" in c.lower() and "clash" in c.lower()]
    if prot_clash_cols:
        no_clash_cols = [c for c in bool_cols if c not in prot_clash_cols]
        all_pass_no_clash = bust_results[no_clash_cols].all(axis=1)
        n_nc = all_pass_no_clash.sum()
        rate_nc = n_nc / len(bust_results) * 100
        print(f"  {'PB-Valid (excl. prot-lig clash)':45s}  {n_nc:5d}/{len(bust_results):5d}  ({rate_nc:5.1f}%)")
 
    print("=" * 70)
 
    # ── Save summary ──
    summary_rows.append({"test": "PB-Valid (ALL)", "n_pass": n_all,
                         "n_total": len(bust_results), "pass_rate": rate_all})
    if prot_clash_cols:
        summary_rows.append({"test": "PB-Valid (excl. prot-lig clash)", "n_pass": n_nc,
                             "n_total": len(bust_results), "pass_rate": rate_nc})
 
    summary_csv = os.path.join(args.outdir, "posebusters_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
 
 
if __name__ == "__main__":
    main()