# Quantum-Compatible AutoDock Grid Energy Evaluation for Multi-Receptor, Multi-Ligand, and Multi-Pose Protein–Ligand Scoring

Author: Pei-Kun Yang

This repository contains the workflow, scripts, datasets, and analysis codes associated with the study:

Pei-Kun Yang. Quantum-Compatible AutoDock Grid Energy Evaluation for Multi-Receptor, Multi-Ligand, and Multi-Pose Protein–Ligand Scoring.
📧 E-mail: [peikun@isu.edu.tw](mailto:peikun@isu.edu.tw)  
🆔 ORCID: [0000-0003-1840-6204](https://orcid.org/0000-0003-1840-6204)

## Workflow

### 1. Select protein-ligand complexes

Protein-ligand complexes are selected and prepared for grid-based scoring. The workflow begins from protein-ligand complex structures, filters the systems according to ligand size and atom-type constraints, and prepares the selected complexes for later molecular dynamics simulation and grid-map generation.

### 2. Molecular dynamics simulation

Molecular dynamics simulations are performed to generate relaxed protein-ligand complex structures and ligand-free protein conformations. These MD-derived conformations are used as receptor structures for subsequent grid-based energy evaluation.

### 3. Generate maps and occupancy grids, then calculate classical energies

AutoDock-style receptor energy maps are generated for the selected protein conformations. Ligand charge grids and atom-type occupancy grids are constructed on the same Cartesian grid. Classical interaction energies are calculated by direct dot products between receptor maps and ligand grids.

### 4. Mimic quantum-circuit energy calculation

The same grid-based energy evaluation is reformulated as a quantum-compatible inner-product calculation. Receptor maps and ligand grids are encoded as normalized vectors, and probability-derived energy recovery is used to mimic the Hadamard-test-based quantum readout. The workflow also evaluates finite-shot effects and map-truncation effects.

### 5. Prepare results for publication

The scripts and generated outputs support figure preparation, numerical analysis, validation, and manuscript preparation for publication. The repository is intended to provide the reference implementation and supporting workflow for the QSBVS study.

## Repository structure

```text
1_data/        Input structures, selected complexes, and prepared molecular files
2_MD/          Molecular dynamics preparation and simulation workflow
3_Autodock/    AutoDock-related preparation and map generation
4_Energy/      Classical and quantum-compatible energy calculations
5_paper/       Analysis outputs, figures, and publication-related files
```

## License

This project is released under the MIT License.
