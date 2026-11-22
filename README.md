# Vina scoring of co-folded protein–ligand complexes

Given a protein–ligand complex predicted by **AlphaFold3, OpenFold3 or Boltz**, this
computes an **AutoDock Vina binding score** for the predicted pose. It also ships as a
self-contained Apptainer container that can **co-fold a complex with OpenFold3 first**
and then score it, so the whole path from a sequence plus a SMILES to a binding score
is one command.

It is a *rescoring* pipeline, not a blind docking one: the search box is taken from
where the predictor placed the ligand, so it answers *"how good is this pose"* rather
than *"where does this ligand bind"*. Blind redocking is available with `--dock`.

---

## Why not just read the ligand out of the CIF

CIF and PDB files store atoms and coordinates but **no bond orders**. If you load the
ligand without telling the toolkit what the molecule actually is, every double bond
comes back single and the hydrogens are added to match — glutathione turns into
`C10H25N3O6S` instead of `C10H17N3O6S`, with all three carbonyls read as alcohols.
Vina then scores a molecule with the wrong H-bond donors, acceptors and rotatable
bonds.

This pipeline rebuilds the ligand's chemistry from its SMILES, which is found
automatically:

| predictor | where the SMILES is | found by |
|---|---|---|
| AlphaFold3 | in the CIF itself (`_chem_comp.pdbx_smiles`) | read directly |
| OpenFold3 | `inference_query_set.json` two directories above the CIF, or the query JSON | searched upward |
| Boltz | `<job>.yaml` above `boltz_results_*/` | searched upward |

The predicted coordinates are kept untouched; only bond orders, formal charges and
hydrogens are restored. An input file is only trusted if it belongs to the same job
as the CIF (by name), so a stray input file nearby cannot redefine the molecule.

**Check the printed formula on every run** — it is the cheapest proof the right
molecule was scored.

---

## The container

```bash
apptainer pull oras://ghcr.io/singhnitink/docking1/docking-of3:latest
```

or build it yourself:

```bash
# with the 2.3 GB OpenFold3 checkpoint baked in:
apptainer build --build-arg BAKE_WEIGHTS=yes docking_of3.sif docking_of3.def
```

### Mode 1 — fold, then score

```bash
apptainer run --nv docking_of3.sif --name gsh \
    --fasta MNPISRSFSHRIIGSRSLDSVLFSPREARSTFPDLENFRIAKLHGFRRVFAHS... \
    --smiles 'SC[C@@H](C(=O)NCC(=O)O)NC(=O)CC[C@@H](C(=O)O)N' \
    --output-dir results/
```

`--fasta` takes the sequence itself or a path to a FASTA file holding one sequence.
Exactly **one** structure is produced (1 model seed, 1 diffusion sample). Give
`--smiles` several times to co-fold several ligands; each is then scored separately
with the others treated as part of the receptor.

`--nv` is required and must be on the `apptainer` command. MSAs come from the
ColabFold server by default, which needs outbound internet; `--no-msa-server` runs
single-sequence mode (fast, less accurate, no network). The driver checks that the
GPU and the MSA server are reachable *before* loading the 2.3 GB of weights, and
says plainly what is missing — OpenFold3's own client would otherwise retry an
unreachable server forever.

Each fold gets its own scratch directory under `/tmp` (OpenFold3's default is one
shared path per user, which two concurrent runs trip over). It is removed when the
run ends; a run that is killed cannot remove it, so the next run does.

**Parameters.** A slim image has no checkpoint baked in. The first time, fetch it
once (public download, no credentials):

```bash
apptainer run docking_of3.sif --setup-weights
```

It lands in `$OPENFOLD_CACHE` (default `~/.openfold3`, which Apptainer mounts from
the host, so it persists and is shared by every later run). Both `of3-p2-155k.pt`
and `of3-p2-145k.pt` are recognised, as is a `ckpt_root` redirection file. To use a
checkpoint stored elsewhere:

```bash
apptainer run --nv --bind /path/to/weights docking_of3.sif \
    --checkpoint /path/to/weights/of3-p2-155k.pt ...
```

### Mode 2 — score a structure you already have

```bash
apptainer run docking_of3.sif --cif my_model.cif
apptainer run docking_of3.sif --cif af3_output/ --output-dir results/   # whole tree
```

No GPU, no network. AlphaFold3, OpenFold3 and Boltz output all work with no extra
flags. If the SMILES cannot be recovered (no input file nearby), give it explicitly:

```bash
apptainer run docking_of3.sif --cif complex.cif --smiles 'c1ccccc1' --ligand-chain B
```

Extra Vina options go after a bare `--`:

```bash
apptainer run docking_of3.sif --cif model.cif -- --dock --exhaustiveness 32
```

### Paths inside the container

Only `$HOME`, the directory you run from and `/tmp` are visible inside. A structure
or an output directory anywhere else needs `--bind /that/path` on the `apptainer`
command. The driver refuses to write results into the container's own ephemeral
filesystem (they would vanish on exit) and says so; a host `/tmp` that happens to
be a tmpfs is recognised as real and allowed.

Output directories on NFS work, including several hosts scoring into the same one:
each structure's working directory is locked while it is scored, so a second run
is told to wait instead of corrupting the first.

Set `RUN_PIPELINE_TRACE=/some/file` (with `--env` on the `apptainer` command) to
get a timestamped trace of every command the driver runs, for debugging.

---

## Running the scorer directly

```bash
pip install numpy biopython rdkit pyyaml      # plus vina >= 1.2 and openbabel on PATH
python docking.py gsh_model.cif
python docking.py af_output/ -j 8 -o scores.csv
```

Tested with Vina 1.2.5 and Open Babel 3.1.1 (the versions Ubuntu 24.04 ships).

---

## Reading the output

```
--- gsh ---
  ligand          : LIG_K (chain K)  C10H17N3O6S  charge 0  20 heavy atoms
  SMILES          : SC[C@@H](C(=O)NCC(=O)O)NC(=O)CC[C@@H](C(=O)O)N   (from CIF)
  Vina score_only : +2.678 kcal/mol   (pose as predicted)
  Vina local_min  : -2.245 kcal/mol   (relaxed in place, moved 2.65 A)
  ! positive score_only: the predicted pose clashes; trust local_min instead
```

| column | meaning |
|---|---|
| `vina_score_only` | the pose scored exactly as predicted, nothing moved |
| `vina_local_min` | after local minimisation in place; `local_min_rmsd` is how far it moved |
| `dock_best` | with `--dock`: best redocked pose, and its RMSD to the predicted pose |
| `smiles` / `smiles_as_built` | what was requested / what the 3D coordinates actually are |
| `smiles_source` | where the SMILES came from (`CIF`, `--smiles`, or the input file's path) |
| `status` | `ok`, `skipped` (protein-only prediction, lone ion) or `failed` |

A complex with several ligand chains gives one row per chain, named `<job>__<chain>`.
The exit status is non-zero if **any** structure failed; skips do not count.

Three warnings are worth acting on:

- **positive `score_only`** — the predicted pose has a steric clash. `local_min`
  relieves it; a large `local_min_rmsd` (> ~1.5 Å) means the pose needed real
  repair, so treat the prediction as unreliable.
- **`flipped ...`** — the predictor produced the wrong stereoisomer. The scores
  describe the real pose, but `smiles` does not; `smiles_as_built` does. Exclude
  these rows (`stereo_ok` is `False`) when comparing isomer-specific ligands.
- **`unspecified stereocentre(s)`** — the input SMILES left a centre undefined, so
  the predictor picked one arbitrarily and separate runs may disagree. Fix the input
  SMILES and refold.

Results are cached per structure in `<output-dir>/vina_work/`. A cache entry is
reused only when the settings, the CIF's contents **and the ligand definition** are
all unchanged, so editing an input file and re-scoring always recomputes. Two runs
cannot use one working directory at the same time; the second one says so and exits.

### Known limitations

- **Protonation.** The ligand is scored exactly as its SMILES specifies, so it
  matches what the predictor saw. If that is the neutral form of an acid or base,
  the ligand is neutral while the receptor is protonated at `--ph`. Vina ignores
  partial charges, so only H-bond typing is affected — but comparisons *between*
  different ligands are less reliable than comparisons between variants of one.
- **Metal ions in the receptor.** Open Babel's protonation step adds hydrogens to
  bare metal ions (Ca²⁺ becomes CaH₂). The ion stays in place, but its typing is
  wrong; if an ion sits in the binding site, prefer to compare scores between
  structures that all contain it rather than against ones that do not.

---

## How the image is published

Every push to `main` rebuilds the image on GitHub Actions, scores `gsh_model.cif`
inside it to check the ligand still comes out as `C10H17N3O6S`, and publishes it to
GHCR ([workflow](.github/workflows/container.yml)). A manual run can bake the
OpenFold3 weights in (`-withweights` tag).

---

## Structure prediction with AlphaFold3

[`launcher.sh`](launcher.sh) runs AlphaFold3 (via Singularity) on
[`fold_input.json`](fold_input.json) to produce a CIF this pipeline consumes.
[`run_alphafold.py`](run_alphafold.py) is DeepMind's unmodified runner. AlphaFold3
cannot be redistributed, which is why the container ships OpenFold3 instead.
