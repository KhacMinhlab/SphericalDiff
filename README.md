# SphericalDiff

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://e3nn.org/"><img alt="e3nn" src="https://img.shields.io/badge/Equivariance-e3nn-blue"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>
</div>

---

## Overview

**SphericalDiff** is a structure-based drug design (**SBDD**) model that generates chemically valid, geometrically accurate 3D ligand molecules conditioned on target protein binding pockets. The model couples an **Elucidated Diffusion Model (EDM)** framework with a novel **MACE-many-body atomic cluster expansion** backbone (`EGNN_Spherical`), enabling expressive, many-body E(3)-equivariant message passing with separate noise schedules for atomic coordinates and atom-type features.

<div align="center">

> *"Generating 3D drug-like molecules directly in binding pockets via E(3)-equivariant many-body diffusion with spherical harmonic representations."*

<img src="image/Abstract.png" alt="SphericalDiff Architecture Overview" width="800"/> 

<div align="center">
  <video src="https://github.com/user-attachments/assets/8d5e4779-6518-421e-932e-fffd143993d7" 
         autoplay 
         loop 
         muted 
         playsinline 
         preload="auto" 
         width="100%">
  </video>
</div>
</div>

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [CrossDocked Benchmark](#crossdocked-benchmark)
  - [Binding MOAD](#binding-moad)
- [Demo](#demo)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Molecular Metrics](#molecular-metrics)
  - [Docking with QuickVina 2](#docking-with-quickvina-2)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## System Requirements

### OS Requirements

Developed and tested under **Python 3.10.x**. Refer to `environment.yml` for the complete pinned dependency list.

---

## Installation

**Clone and install dependencies**
```bash
git clone https://github.com/KhacMinhlab/SphericalDiff
cd SphericalDiff

conda env create -f environment.yml
conda activate SphericalDiff

```

**Download model checkpoints**

Model weights, processed BindingMOAD testset, filtering report, and associated case-study files are hosted on Zenodo (DOI: 10.5281/zenodo.22054043). 


### QuickVina 2 (for docking evaluation)
```bash
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1
mv qvina2.1 $HOME/mambaforge/envs/SphericalDiff/bin
```

For receptor preparation (PDB → PDBQT), create a separate MGLTools environment to avoid dependency conflicts:
```bash
mamba create -n mgltools -c bioconda mgltools
```

---

## Data Preparation

### CrossDocked Benchmark

Download and extract the dataset following the instructions from [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol/tree/main/data). Then process the raw data:

```bash
python scripts/process_crossdock.py <crossdocked_dir> --no_H
```

The processed dataset will be saved with C-alpha only pocket representations by default (matching the `CA` pocket mode in the config).

### Binding MOAD

**Download the dataset:**
```bash
wget https://zenodo.org/record/<record_id>/files/every_part_a.zip
wget https://zenodo.org/record/<record_id>/files/every_part_b.zip
wget https://zenodo.org/record/<record_id>/files/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```

**Process the raw data:**
```bash
python scripts/process_bindingmoad.py <bindingmoad_dir>

# To suppress RDKit / BioPython warnings:
python -W ignore scripts/process_bindingmoad.py <bindingmoad_dir>
```

---

## Demo

<div align="center">
  <video src="image/Sampling%20Process.mp4"
         autoplay
         loop
         muted
         playsinline
         preload="auto"
         width="100%">
  </video>
</div>

### Generate molecules for a given binding pocket

The generation script supports two pocket-definition modes and an optional two-phase inpainting pipeline (Phase-1 generation, optional Phase-2 fragment growing/refinement).

**By explicit pocket residues:**
```bash
python generate_ligands.py \
    --checkpoint <checkpoint>.ckpt \
    --pdbfile <protein>.pdb \
    --pocket_ids A:1 A:2 A:3 A:4 A:5 A:6 A:7 \
    --outfile results/generated.sdf
```

**By reference ligand (defines the pocket via distance cutoff):**
```bash
python generate_ligands.py \
    --checkpoint <checkpoint>.ckpt \
    --pdbfile <protein>.pdb \
    --ref_ligand <reference_ligand>.sdf \
    --outfile results/generated.sdf
```

**Growing molecules from a fixed anchor fragment (Phase-1 manual inpainting):**
```bash
python generate_ligands.py \
    --checkpoint <checkpoint>.ckpt \
    --pdbfile <protein>.pdb \
    --ref_ligand <reference_ligand>.sdf \
    --fix_atoms <anchor_fragment>.sdf \
    --outfile <output>.sdf \
    --n_samples <N> \
    --n_nodes_min <min_heavy_atoms> \
    --resamplings <r> \
    --sanitize \
    --relax \
    --largest_frag
```
**Required arguments:**

| Flag | Description |
|------|-------------|
| `--checkpoint` | Path to the LigandPocketEDM checkpoint (`.ckpt`). |
| `--pdbfile` | Path to the target protein PDB file. |
| `--outfile` | Output SDF file path for the generated molecules (a file, not a directory). |

**Pocket definition (choose one):**

| Flag | Description |
|------|-------------|
| `--ref_ligand` | SDF path, or `<chain>:<resi>` — defines the pocket via distance cutoff around the reference ligand. |
| `--pocket_ids` | Explicit pocket residues as `<chain>:<resi>` pairs (e.g. `A:100 A:101 A:150`), bypassing the distance-cutoff pocket definition. |

**Phase-1 manual inpainting (optional):**

| Flag | Description |
|------|-------------|
| `--fix_atoms` | A single SDF path → fixes all atoms in that file during generation. Atom names (e.g. `C1 N2`) → fixes those specific atoms from the `--ref_ligand` PDB residue instead. |
| `--add_n_nodes` | Number of extra atoms to generate on top of the fixed substructure. If omitted, the model samples a total size from its learned p(N_lig \| N_pocket) distribution. |

**Sampling configuration (optional):**

| Flag | Description |
|------|-------------|
| `--n_samples` | Target number of valid molecules to collect (default: 100). |
| `--batch_size` | Molecules requested per Phase-1 batch (default: 60). |
| `--num_sampling_steps` | Karras denoising steps (default: 40). |
| `--n_nodes_min` | Minimum atom count for sampled molecule size. |
| `--resamplings` | Repaint r for Phase-1 (default: 1). |
| `--sanitize` | Apply RDKit sanitization to generated molecules. |
| `--relax` | Apply 200-step MMFF relaxation to generated structures. |
| `--largest_frag` | Keep only the largest fragment from each generated molecule. |

**Phase-2 conditional inpainting (optional, enable with `--refine_fragments`):** grows small/disconnected Phase-1 outputs. See `python generate_ligands.py --help` for the full set of Phase-2 and advanced sampling flags (fragment-growing thresholds, repulsion guidance, Karras noise-schedule overrides).

---

## Training

**Start a new training run:**
```bash
python -u train.py --config configs/ca_config_sphericaldiff.yml
```

**Resume from a checkpoint:**
```bash
python -u train.py --config configs/ca_config_sphericaldiff.yml --resume <checkpoint>.ckpt
```

 The reference configuration for the CA-pocket EDM+MACE model is `configs/ca_config_sphericaldiff.yml`. 

---

## Evaluation

### Reproduce Paper Results

To sample molecules for the full test set:
```bash
python test.py <checkpoint>.ckpt \
    --test_dir <bindingmoad_dir>/processed_noH/test/ \
    --outdir <output_dir> \
    --fix_n_nodes
```

The `--fix_n_nodes` flag constrains the sampled molecules to have the same number of heavy atoms as the reference ligand (used for fair comparison in benchmarks).

### Molecular Metrics

To compute QED, SA, LogP, Lipinski, and diversity scores, use `eval_ligands.py`:

```bash
python eval_ligands.py <input_directory> <output_directory>
```

`<input_directory>` is scanned recursively for `.sdf` files (one file per pocket); results are written to a `metrics_<uuid>.csv` file in `<output_directory>` with per-molecule QED, SA, LogP, Lipinski, and per-pocket diversity.

Equivalently, from Python:
```python
from analysis.metrics import MoleculeProperties

mol_metrics = MoleculeProperties()
all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
    mol_metrics.evaluate(pocket_mols)
```

`evaluate()` accepts a list of lists, where each inner list holds all RDKit `Mol` objects generated for a single pocket.

### Docking with QuickVina 2

**Step 1 — Prepare receptor files (PDB → PDBQT):**
```bash
conda activate mgltools
cd analysis
python2 docking_py27.py \
    <bindingmoad_dir>/processed_noH/test/ \
    <docking_outdir> \
    bindingmoad
cd ..
conda deactivate
```

**Step 2 — Run docking:**
```bash
conda activate SphericalDiff
python3 analysis/docking.py \
    --pdbqt_dir <docking_outdir> \
    --sdf_dir <test_outdir> \
    --out_dir <qvina_outdir> \
    --write_csv \
    --write_dict \
    --dataset moad
```

> Downstream molecule analysis scripts are available in `analysis/inference_analysis.py` and `analysis/molecule_analysis.py`.

---

## Acknowledgements

SphericalDiff is built upon and inspired by the following outstanding works:

- [GCDM-SBDD](https://github.com/BioinfoMachineLearning/GCDM-SBDD) — Geometry-Complete Diffusion for SBDD
- [DiffSBDD](https://github.com/arneschneuing/DiffSBDD) — Diffusion-based structure-based drug design
- [MACE](https://github.com/ACEsuit/mace) — Many-body equivariant message passing
- [e3nn](https://github.com/e3nn/e3nn) — Euclidean neural networks library
- [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol) — Pocket-conditioned molecule generation
- [EDM (Karras et al.)](https://arxiv.org/abs/2206.00364) — Elucidating the Design Space of Diffusion-Based Generative Models

We sincerely thank all their contributors and maintainers.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

**Under this license, you are free to:**
* **Share**: Copy and redistribute the material in any medium or format.
* **Adapt**: Remix, transform, and build upon the material.

**Under the following terms:**
* **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **NonCommercial**: You may not use the material for commercial purposes.
* **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

See the [LICENSE](LICENSE.md) file for the full legal text.
---

## Citation

If you use SphericalDiff in your research, please cite:
