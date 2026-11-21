#!/usr/bin/env python
"""
AutoDock Vina scoring of ligand poses from co-folded structures
(AlphaFold3, OpenFold3 or Boltz).

For each CIF this:

  1. finds the ligand chain(s) and the ligand SMILES. AlphaFold3 stores the SMILES
     in the CIF itself; OpenFold3 and Boltz keep it in the input JSON / YAML that
     sits beside or above the output, which is searched for automatically,
  2. splits the complex into receptor (everything except the ligand chain) and ligand,
  3. rebuilds the ligand with correct bond orders / formal charges using that SMILES
     as a template.  THIS IS THE STEP THAT MATTERS: a CIF/PDB stores no bond orders,
     so reading the ligand without a template turns every C=O into C-OH and silently
     scores the wrong molecule.  The predicted coordinates are kept exactly; only
     chemistry is restored.  Predicted stereochemistry is checked against the SMILES,
  4. prepares PDBQT files (Open Babel; receptor protonated at --ph),
  5. runs Vina:
       score_only : score of the pose as predicted (no movement)
       local_only : score after local minimisation in place (+ RMSD moved)
       dock       : optional (--dock) redock in a box around the predicted pose,
                    reporting best score and each pose's RMSD to the predicted pose,
  6. prints a summary and (for >1 structure, or with -o) writes one CSV row each.
     A complex with several ligand chains gets one row per chain, the other
     ligands being treated as part of the receptor.

Usage
  python docking.py gsh_model.cif
  python docking.py gsh_model.cif --dock --keep_files
  python docking.py af_output/ -o scores.csv -j 8

Requires: rdkit, biopython, numpy (pyyaml for Boltz input files);
`vina` (>=1.2) and `obabel` on PATH.

Note on protonation: the ligand is built exactly as the folded SMILES specifies, so
it is scored in the same protonation state the predictor saw.  If that SMILES is the
neutral form of an acid or base, the ligand is scored neutral while the receptor is
protonated at --ph.  Vina's scoring function ignores partial charges, so this only
affects H-bond donor/acceptor typing -- but keep it in mind when comparing different
ligands to each other.
"""

import argparse
import csv
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import string
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds

RDLogger.DisableLog("rdApp.*")

VINA = "vina"
OBABEL = "obabel"

# AutoDock atom type -> element
AD2ELEM = {"A": "C", "C": "C", "OA": "O", "O": "O", "NA": "N", "N": "N", "NS": "N",
           "SA": "S", "S": "S", "HD": "H", "H": "H", "HS": "H", "F": "F", "Cl": "Cl",
           "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I", "P": "P", "MG": "Mg",
           "ZN": "Zn", "MN": "Mn", "CA": "Ca", "FE": "Fe"}

CSV_FIELDS = [
    "name", "status", "error",
    "vina_score_only", "vina_local_min", "local_min_rmsd",
    "dock_best", "dock_best_rmsd", "dock_best_within_2A", "dock_rmsd_of_best_within_2A",
    "n_heavy", "formula", "formal_charge", "stereo_ok", "stereo_note",
    "ligand_chain", "ligand_comp", "smiles", "smiles_as_built", "smiles_source",
    "af3_ranking_score", "af3_ptm", "af3_iptm", "af3_ligand_chain_iptm", "af3_has_clash",
    "cif", "workdir",
]

BIND_HINT = ("inside a container only $HOME, the working directory and /tmp are "
             "visible by default; anything else needs --bind <path>")


def warn(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# input discovery
# --------------------------------------------------------------------------------------
def find_cifs(inputs):
    """Every *.cif under each directory. AlphaFold3 writes <job>_model.cif, Boltz
    <job>_model_N.cif and OpenFold3 <query>_seed_N_sample_M_model.cif; a file that is
    not a structure is rejected later, loudly, rather than skipped here, silently."""
    cifs = []
    for p in inputs:
        if os.path.isfile(p):
            cifs.append(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if f.lower().endswith(".cif"):
                        cifs.append(os.path.join(root, f))
        else:
            sys.exit(f"no such file or directory: {p}\n  ({BIND_HINT})")
    return sorted(set(os.path.abspath(c) for c in cifs))


def job_name(cif):
    """A label unique to the file: gsh_model.cif -> gsh; A208K_model_0.cif -> A208K_model_0;
    seed-1_sample-0/model.cif -> <job>_seed-1_sample-0."""
    base = os.path.basename(cif)
    if base == "model.cif":
        d = os.path.basename(os.path.dirname(cif))
        parent = os.path.basename(os.path.dirname(os.path.dirname(cif)))
        return f"{parent}_{d}" if parent else d
    return re.sub(r"_model$", "", os.path.splitext(base)[0])


def base_name(cif):
    """The job / query the file belongs to, every predictor's suffix stripped:
    gsh_model.cif -> gsh;  A208K_model_0.cif -> A208K;
    alginate_seed_42_sample_1_model.cif -> alginate;  <job>/seed-1_sample-0/model.cif -> <job>."""
    n = job_name(cif)
    n = re.sub(r"_model(_\d+)?$", "", n)
    n = re.sub(r"_seed[-_]\d+_sample[-_]\d+$", "", n)
    return n


def workdir_tag(cif):
    """Readable, but unique per *file*: two structures that share a basename must
    never share a working directory, or they overwrite each other's PDBQT files."""
    h = hashlib.sha1(os.path.abspath(cif).encode()).hexdigest()[:6]
    return f"{job_name(cif)}-{h}"


def placeholder_ancestors(path):
    """Directories above the CIF's directory whose ONLY entry is the next step of
    the path. That is what Apptainer leaves when it binds just the working
    directory: the path to it exists, but every level above the leaf is a bare
    placeholder, so an input file one level up is invisible inside the container."""
    out = []
    child = os.path.dirname(os.path.abspath(path))
    d = os.path.dirname(child)
    for _ in range(6):
        if d == child:
            break
        try:
            entries = os.listdir(d)
        except OSError:
            break
        if entries != [os.path.basename(child)]:
            break
        out.append(d)
        child, d = d, os.path.dirname(d)
    return out


# --------------------------------------------------------------------------------------
# ligand identity
# --------------------------------------------------------------------------------------
def _as_list(d, key):
    v = d.get(key, [])
    return v if isinstance(v, list) else [v]


def read_cif(cif):
    """Parse an AlphaFold3 / OpenFold3 / Boltz CIF -> (first model, mmcif dict).

    OpenFold3 omits _atom_site.occupancy, which Biopython's MMCIFParser requires,
    so fill in the columns it assumes before handing the dict over.
    """
    d = MMCIF2Dict(cif)
    n = len(_as_list(d, "_atom_site.id"))
    if not n:
        raise ValueError(f"no _atom_site records in {cif} (not a structure file?)")
    for key, default in (("_atom_site.occupancy", "1.00"),
                         ("_atom_site.label_alt_id", "."),
                         ("_atom_site.pdbx_PDB_model_num", "1"),
                         ("_atom_site.B_iso_or_equiv", "0.00")):
        if key not in d:
            d[key] = [default] * n
    p = MMCIFParser(QUIET=True)
    p._mmcif_dict = d
    p._build_structure("c")
    return next(iter(p._structure_builder.get_structure())), d


def ligand_from_cif(d):
    """-> list of (comp_id, chain, smiles or None) for every non-polymer component.

    AlphaFold3 stores the ligand SMILES in the CIF (_chem_comp.pdbx_smiles).
    OpenFold3 and Boltz do not, but do mark the ligand chain via _entity /
    _struct_asym, so the chain is found here and the SMILES comes from the input file.
    """
    ids = _as_list(d, "_chem_comp.id")
    smis = _as_list(d, "_chem_comp.pdbx_smiles")
    typs = _as_list(d, "_chem_comp.type")
    smap = {}
    for i, c in enumerate(ids):
        t = typs[i].lower() if i < len(typs) else ""
        s = smis[i] if i < len(smis) else "?"
        if "non-polymer" in t and s not in ("?", ".", ""):
            smap[c] = s

    ent_type = dict(zip(_as_list(d, "_entity.id"), _as_list(d, "_entity.type")))
    nonpoly_chains = {a for a, e in zip(_as_list(d, "_struct_asym.id"),
                                        _as_list(d, "_struct_asym.entity_id"))
                      if "non-polymer" in ent_type.get(e, "").lower()}
    nonpoly_chains |= set(_as_list(d, "_pdbx_nonpoly_scheme.asym_id"))

    comps = _as_list(d, "_atom_site.label_comp_id")
    labs = _as_list(d, "_atom_site.label_asym_id")
    auths = _as_list(d, "_atom_site.auth_asym_id") or labs
    out, seen = [], set()
    for c, lab, ch in zip(comps, labs, auths):
        if (c in smap or lab in nonpoly_chains or ch in nonpoly_chains) and (c, ch) not in seen:
            seen.add((c, ch))
            out.append((c, ch, smap.get(c)))
    return out


def _source_name(j, path):
    """The job a predictor input file belongs to: its "name" field if it has one,
    else the file's stem (Boltz: A208K.yaml; AlphaFold3: gsh_data.json)."""
    n = j.get("name")
    if isinstance(n, str) and n.strip():
        return n.strip()
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_data$", "", stem)


def _smiles_from_input_obj(j, path, chain, base):
    """Pull the ligand SMILES for `chain` out of a predictor input file, but ONLY
    if that file belongs to the same job as the CIF. Chain "B" is the default
    ligand chain for every predictor, so without this check any stray input file
    nearby would silently redefine the molecule being scored."""
    # AlphaFold3 / Boltz: {"name": ..., "sequences": [{"ligand": {"id": [...], "smiles": ...}}]}
    seqs = j.get("sequences")
    if isinstance(seqs, list) and _source_name(j, path).lower() == base.lower():
        for s in seqs:
            lig = s.get("ligand") if isinstance(s, dict) else None
            if not lig or not lig.get("smiles"):
                continue
            ids = lig["id"] if isinstance(lig["id"], list) else [lig["id"]]
            if chain is None or chain in ids:
                return lig["smiles"]
    # OpenFold3: {"queries": {"<name>": {"chains": [{"molecule_type": "ligand",
    #             "chain_ids": [...], "smiles": ...}]}}}  -- the key is the job name
    queries = j.get("queries") or {}
    for qn, q in queries.items():
        if qn != base:
            continue
        for ch in (q or {}).get("chains", []) or []:
            # OpenFold3's own inference_query_set.json spells this "LIGAND"
            if (ch.get("molecule_type") or "").lower() != "ligand" or not ch.get("smiles"):
                continue
            ids = ch.get("chain_ids") or []
            if chain is None or chain in ids or not ids:
                return ch["smiles"]
    return None


def _load_input_file(path):
    """Load an AlphaFold3/OpenFold3 JSON or a Boltz YAML into a dict."""
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            return None
        with open(path) as fh:
            return yaml.safe_load(fh)
    with open(path) as fh:
        return json.load(fh)


def smiles_from_input(cif, chain):
    """Search beside and above the CIF for the input file that defined this ligand.
    -> (smiles, path) or (None, None).

    AlphaFold3: <job>_data.json / fold_input.json beside the CIF.
    OpenFold3:  inference_query_set.json two levels up, or the query JSON given to
                run_openfold a few levels up.
    Boltz:      <job>.yaml above boltz_results_*/.
    """
    base = base_name(cif)
    seen = set()
    d = os.path.dirname(os.path.abspath(cif))
    for _ in range(6):
        cands = []
        for pat in ("*.json", "*.yaml", "*.yml"):
            cands += sorted(glob.glob(os.path.join(d, pat)))
        for sub in sorted(glob.glob(os.path.join(d, "*json*"))):
            if os.path.isdir(sub):
                cands += sorted(glob.glob(os.path.join(sub, "*.json")))
        for p in cands:
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            try:
                if os.path.getsize(p) > 512 * 1024 * 1024:
                    continue
                j = _load_input_file(p)
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            s = _smiles_from_input_obj(j, p, chain, base)
            if s:
                return s, p
        d = os.path.dirname(d)
    return None, None


def af3_metadata(cif, chain):
    """<job>_summary_confidences.json, if AlphaFold3 wrote one next to the CIF."""
    d = os.path.dirname(os.path.abspath(cif))
    for cand in (f"{job_name(cif)}_summary_confidences.json", "summary_confidences.json"):
        p = os.path.join(d, cand)
        if not os.path.isfile(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            return {}
        out = {"af3_ranking_score": j.get("ranking_score"), "af3_ptm": j.get("ptm"),
               "af3_iptm": j.get("iptm"), "af3_has_clash": j.get("has_clash")}
        ids, ci = j.get("chain_ids"), j.get("chain_iptm")
        if ids and ci and chain in ids:
            out["af3_ligand_chain_iptm"] = ci[ids.index(chain)]
        return out
    return {}


# --------------------------------------------------------------------------------------
# structure preparation
# --------------------------------------------------------------------------------------
class SkipLigand(Exception):
    """Not a failure: nothing here that Vina can meaningfully score."""


class ReceptorSelect(Select):
    """Everything except the ligand chain and solvent."""

    def __init__(self, lig_chain):
        self.lig_chain = lig_chain

    def accept_chain(self, chain):
        return chain.id != self.lig_chain

    def accept_residue(self, res):
        return res.get_resname().strip() not in ("HOH", "WAT", "DOD")


def split_complex(model, lig_chain, receptor_pdb):
    """Write the receptor PDB; return the ligand's heavy-atom elements and coordinates."""
    if lig_chain not in model:
        raise ValueError(f"ligand chain {lig_chain} not in CIF (chains: {[c.id for c in model]})")

    # PDB format allows only 1-character chain IDs; AlphaFold3 can emit longer ones.
    longs = [c for c in model if len(c.id.strip()) > 1 and c.id != lig_chain]
    if longs:
        free = [x for x in string.ascii_uppercase + string.digits
                if x not in {c.id for c in model}]
        for c in longs:
            if not free:
                raise ValueError("too many chains to remap into PDB single-character IDs")
            c.id = free.pop(0)

    io = PDBIO()
    io.set_structure(model)
    io.save(receptor_pdb, ReceptorSelect(lig_chain))

    elems, xyz = [], []
    for atom in model[lig_chain].get_atoms():
        e = (atom.element or atom.get_id()[0]).strip().capitalize()
        if e in ("H", "D"):
            continue
        elems.append(e)
        xyz.append(atom.coord.astype(float))
    if not elems:
        raise ValueError(f"no atoms in ligand chain {lig_chain}")
    return elems, np.array(xyz)


def mol_from_xyz(elems, xyz):
    rw = Chem.RWMol()
    for e in elems:
        rw.AddAtom(Chem.Atom(e))
    conf = Chem.Conformer(len(elems))
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, p.tolist())
    rw.AddConformer(conf, assignId=True)
    m = rw.GetMol()
    rdDetermineBonds.DetermineConnectivity(m)
    # DetermineConnectivity marks atoms NoImplicit; clear so H counts follow template bond orders
    for a in m.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumRadicalElectrons(0)
    m.UpdatePropertyCache(strict=False)
    return m


def template_from_smiles(smiles):
    t = Chem.MolFromSmiles(smiles)
    if t is None:
        raise ValueError(f"RDKit cannot parse SMILES {smiles}")
    frags = Chem.GetMolFrags(t, asMols=True)
    note = ""
    if len(frags) > 1:  # keep largest fragment (drops counter-ions)
        t = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        note = "multi-fragment SMILES: kept largest fragment"
    return t, note


def build_ligand(elems, xyz, smiles):
    """Restore bond orders/charges/hydrogens onto the predicted coordinates.
    -> (mol with H, stereo_ok, note, smiles_as_built)"""
    template, note = template_from_smiles(smiles)
    probe = mol_from_xyz(elems, xyz)

    # if the CIF ligand chain has extra fragments (counter-ions), keep the biggest
    if len(Chem.GetMolFrags(probe)) > 1:
        frags = Chem.GetMolFrags(probe, asMols=True, sanitizeFrags=False)
        probe = max(frags, key=lambda f: f.GetNumAtoms())
    if probe.GetNumAtoms() != template.GetNumHeavyAtoms():
        raise ValueError(f"heavy atom count mismatch: CIF {probe.GetNumAtoms()} vs SMILES "
                         f"{template.GetNumHeavyAtoms()} (bad geometry or wrong SMILES)")
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, probe)
    except Exception as e:
        raise ValueError(f"could not map SMILES onto predicted ligand graph "
                         f"(distorted geometry?): {e}")
    Chem.SanitizeMol(mol)
    if Chem.MolToSmiles(mol, isomericSmiles=False) != Chem.MolToSmiles(template, isomericSmiles=False):
        raise ValueError("ligand chemistry after bond-order assignment differs from SMILES")

    # stereo check: compare only centres actually specified in the SMILES
    Chem.AssignStereochemistryFrom3D(mol)
    match = mol.GetSubstructMatch(template)
    want = dict(Chem.FindMolChiralCenters(template, includeUnassigned=False,
                                          useLegacyImplementation=False))
    got = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                         useLegacyImplementation=False))
    bad = [f"{i}:{lab}->{got.get(match[i], '?')}" for i, lab in want.items()
           if got.get(match[i]) != lab]
    unspec = [i for i, lab in Chem.FindMolChiralCenters(template, includeUnassigned=True,
                                                        useLegacyImplementation=False) if lab == "?"]
    notes = [note]
    if bad:
        notes.append("flipped " + ",".join(bad))
    if unspec:
        notes.append(f"unspecified stereocentre(s) in SMILES at {unspec}: "
                     f"folded as an arbitrary isomer")
    # what was actually scored, stereo taken from the 3D coordinates
    as_built = Chem.MolToSmiles(mol)
    return Chem.AddHs(mol, addCoords=True), (not bad), "; ".join(filter(None, notes)), as_built


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout + p.stderr


def prep_receptor(receptor_pdb, out_pdbqt, ph, cwd):
    run([OBABEL, receptor_pdb, "-O", out_pdbqt, "-xr", "-p", str(ph)], cwd)
    if not os.path.getsize(os.path.join(cwd, out_pdbqt)):
        raise RuntimeError("empty receptor pdbqt")


def prep_ligand(sdf, out_pdbqt, cwd):
    # hydrogens/charges already set from the SMILES; obabel only types atoms + Gasteiger charges
    run([OBABEL, sdf, "-O", out_pdbqt, "--partialcharge", "gasteiger"], cwd)
    if not os.path.getsize(os.path.join(cwd, out_pdbqt)):
        raise RuntimeError("empty ligand pdbqt")


# --------------------------------------------------------------------------------------
# Vina + RMSD
# --------------------------------------------------------------------------------------
ENERGY_RE = re.compile(r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.\d+)")


def vina_energy(out):
    m = ENERGY_RE.search(out)
    if not m:
        raise RuntimeError(f"could not parse Vina energy:\n{out[-1500:]}")
    return float(m.group(1))


def read_pdbqt_models(path):
    """list of (affinity or None, [(elem, xyz), ...]) per MODEL, atom order preserved."""
    models, cur, aff = [], [], None
    for line in open(path):
        if line.startswith("MODEL"):
            cur, aff = [], None
        elif line.startswith("REMARK VINA RESULT"):
            aff = float(line.split()[3])
        elif line.startswith(("ATOM", "HETATM")):
            t = line[77:79].strip()
            cur.append((AD2ELEM.get(t.upper(), t),
                        np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
        elif line.startswith("ENDMDL"):
            models.append((aff, cur))
    if not models and cur:
        models.append((aff, cur))
    return models


class PoseRMSD:
    """Symmetry-aware heavy-atom RMSD (no superposition) to the predicted pose."""

    def __init__(self, molH, input_pdbqt):
        heavy = Chem.RemoveHs(molH)
        self.ref = heavy.GetConformer().GetPositions()
        self.autos = heavy.GetSubstructMatches(heavy, uniquify=False, useChirality=False,
                                               maxMatches=100000)
        # map pdbqt heavy-atom order -> rdkit heavy-atom index via identical input coordinates
        atoms = [a for a in read_pdbqt_models(input_pdbqt)[0][1] if a[0] != "H"]
        if len(atoms) != len(self.ref):
            raise RuntimeError(f"pdbqt heavy atoms {len(atoms)} != ligand {len(self.ref)}")
        self.order = []
        for _, p in atoms:
            d = np.linalg.norm(self.ref - p, axis=1)
            j = int(np.argmin(d))
            if d[j] > 0.05:
                raise RuntimeError("could not map pdbqt atoms to ligand atoms")
            self.order.append(j)
        if len(set(self.order)) != len(self.order):
            raise RuntimeError("non-unique pdbqt->ligand atom mapping")

    def __call__(self, pose_atoms):
        heavy = [p for e, p in pose_atoms if e != "H"]
        pos = np.empty_like(self.ref)
        for k, j in enumerate(self.order):
            pos[j] = heavy[k]
        # automorphism t maps ref atom i -> equivalent atom t[i]
        return min(float(np.sqrt(np.mean(np.sum((self.ref - pos[list(t)]) ** 2, axis=1))))
                   for t in self.autos)


# --------------------------------------------------------------------------------------
# per-structure job
# --------------------------------------------------------------------------------------
def _file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_chain(cif, wd, a, model, chain, comp, smiles, source, cif_sha, name):
    """Score one ligand chain of one structure. Returns the result row."""
    row = {"name": name, "cif": cif, "workdir": wd, "ligand_chain": chain,
           "ligand_comp": comp, "smiles": smiles, "smiles_source": source}
    # Cache key: every setting that changes the numbers, the CIF's content, AND the
    # ligand definition. Fixing a protonation state or a stereocentre in the input
    # file and re-scoring is a routine workflow; it must never return stale numbers.
    key = hashlib.sha1(f"{a['_args_key']}|{cif_sha}|{smiles}|{chain}".encode()).hexdigest()
    result_json = os.path.join(wd, "result.json")
    if os.path.isfile(result_json) and not a["overwrite"]:
        try:
            cached = json.load(open(result_json))
        except Exception:
            cached = {}
        if cached.get("_key") == key:
            return cached
    os.makedirs(wd, exist_ok=True)

    # One run per working directory at a time: a second process scoring the same
    # structure into the same place would overwrite the PDBQT files mid-run and
    # both would report mixed-up numbers.
    lock = open(os.path.join(wd, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        row.update(status="failed",
                   error=f"another run is scoring this structure into {wd} right now; "
                         f"wait for it, or use a different --workdir/--output-dir")
        return row

    try:
        row.update(af3_metadata(cif, chain))
        elems, xyz = split_complex(model, chain, os.path.join(wd, "receptor.pdb"))
        molH, stereo_ok, stereo_note, as_built = build_ligand(elems, xyz, smiles)
        row.update(n_heavy=molH.GetNumHeavyAtoms(),
                   formula=Chem.rdMolDescriptors.CalcMolFormula(molH),
                   formal_charge=Chem.GetFormalCharge(molH),
                   stereo_ok=stereo_ok, stereo_note=stereo_note,
                   smiles_as_built=as_built)
        if molH.GetNumHeavyAtoms() < 2:
            raise SkipLigand("monatomic ligand (a lone ion) - Vina has no scoring "
                             "terms for it, and Open Babel cannot build a PDBQT")
        w = Chem.SDWriter(os.path.join(wd, "ligand_af3.sdf"))
        w.write(molH)
        w.close()

        prep_receptor("receptor.pdb", "receptor.pdbqt", a["ph"], wd)
        prep_ligand("ligand_af3.sdf", "ligand.pdbqt", wd)
        rmsd = PoseRMSD(molH, os.path.join(wd, "ligand.pdbqt"))

        base = [VINA, "--receptor", "receptor.pdbqt", "--ligand", "ligand.pdbqt",
                "--cpu", str(a["cpu"]), "--seed", str(a["seed"])]

        row["vina_score_only"] = vina_energy(run(base + ["--score_only", "--autobox"], wd))

        out = run(base + ["--local_only", "--autobox", "--out", "ligand_localmin.pdbqt"], wd)
        row["vina_local_min"] = vina_energy(out)
        row["local_min_rmsd"] = round(
            rmsd(read_pdbqt_models(os.path.join(wd, "ligand_localmin.pdbqt"))[0][1]), 3)

        if a["dock"]:
            center = xyz.mean(axis=0)
            size = np.maximum(xyz.max(0) - xyz.min(0) + 2 * a["padding"], 10.0)
            box = []
            for k, ax in enumerate("xyz"):
                box += [f"--center_{ax}", f"{center[k]:.3f}", f"--size_{ax}", f"{size[k]:.3f}"]
            run(base + box + ["--exhaustiveness", str(a["exhaustiveness"]),
                              "--num_modes", str(a["num_modes"]), "--energy_range", "4",
                              "--out", "ligand_docked.pdbqt"], wd)
            poses = read_pdbqt_models(os.path.join(wd, "ligand_docked.pdbqt"))
            scored = [(aff, rmsd(atoms)) for aff, atoms in poses]
            with open(os.path.join(wd, "docked_poses.tsv"), "w") as fh:
                fh.write("mode\tvina\trmsd_to_predicted\n")
                for i, (aff, r) in enumerate(scored, 1):
                    fh.write(f"{i}\t{aff}\t{r:.3f}\n")
            row["dock_best"], r0 = scored[0]
            row["dock_best_rmsd"] = round(r0, 3)
            near = [(aff, r) for aff, r in scored if r <= 2.0]
            if near:
                row["dock_best_within_2A"], rn = near[0]
                row["dock_rmsd_of_best_within_2A"] = round(rn, 3)
        row["status"] = "ok"
    except SkipLigand as e:
        row["status"] = "skipped"
        row["error"] = str(e)
    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e).strip().splitlines()[0][:300] if str(e).strip() else repr(e)
        with open(os.path.join(wd, "error.log"), "w") as fh:
            fh.write(traceback.format_exc())
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    if not a["keep_files"]:
        p = os.path.join(wd, "receptor.pdb")
        if os.path.exists(p):
            os.remove(p)
    if row.get("status") in ("ok", "skipped"):
        json.dump(dict(row, _key=key), open(result_json, "w"), indent=1, default=str)
    return row


def process(job):
    """Score every ligand chain of one CIF. Returns a list of rows."""
    cif, wd, a = job["cif"], job["workdir"], job["args"]
    name = job_name(cif)
    failed = {"name": name, "cif": cif, "workdir": wd, "status": "failed"}

    try:
        model, cifdict = read_cif(cif)
    except Exception as e:
        failed["error"] = f"cannot read as a structure: {str(e).strip().splitlines()[0][:200]}"
        return [failed]

    ligs = ligand_from_cif(cifdict)
    if a["ligand_chain"]:
        ligs = [l for l in ligs if l[1] == a["ligand_chain"]]
        if not ligs:
            model_chains = [c.id for c in model]
            failed["error"] = (f"chain {a['ligand_chain']} is not a ligand chain in this CIF "
                               f"(chains present: {model_chains})")
            return [failed]
    chains = sorted({ch for _, ch, _ in ligs})
    if not chains:
        return [dict(failed, status="skipped",
                     error="no ligand in this structure (protein-only prediction)")]
    if a["smiles"] and len(chains) > 1:
        failed["error"] = (f"--smiles was given but the CIF has several ligand chains "
                           f"{chains}; say which one with --ligand_chain")
        return [failed]

    cif_sha = _file_sha1(cif)
    rows = []
    for chain in chains:
        comp = next(c for c, ch, _ in ligs if ch == chain)
        smiles = a["smiles"] or next(s for _, ch, s in ligs if ch == chain)
        source = "--smiles" if a["smiles"] else ("CIF" if smiles else None)
        if not smiles:
            smiles, source = smiles_from_input(cif, chain)
        rname = name if len(chains) == 1 else f"{name}__{chain}"
        cwd = wd if len(chains) == 1 else os.path.join(wd, f"chain_{chain}")
        if not smiles:
            msg = (f"no SMILES for ligand chain {chain} in the CIF or in any input "
                   f"JSON/YAML beside or above it; pass --smiles")
            ph = placeholder_ancestors(cif)
            if len(ph) >= 2:
                msg += (f". Every directory above the CIF ({len(ph)} levels checked) holds "
                        f"nothing but the path down to it, which is what Apptainer shows when "
                        f"that filesystem is not bound: the input file that defines this "
                        f"ligand is probably there on the host but invisible here. {BIND_HINT}")
            rows.append(dict(failed, name=rname, workdir=cwd, ligand_chain=chain,
                             ligand_comp=comp, error=msg))
            continue
        rows.append(_score_chain(cif, cwd, a, model, chain, comp, smiles, source,
                                 cif_sha, rname))
    return rows


def report(row):
    """Human-readable summary for a single structure."""
    n = row["name"]
    if row.get("status") == "skipped":
        print(f"\n--- {n}: skipped ---\n  {row.get('error')}")
        return
    if row.get("status") != "ok":
        warn(f"\n--- {n}: FAILED ---\n  {row.get('error')}")
        if os.path.isfile(os.path.join(row.get("workdir", ""), "error.log")):
            warn(f"  see {row['workdir']}/error.log")
        return
    print(f"\n--- {n} ---")
    print(f"  ligand          : {row['ligand_comp']} (chain {row['ligand_chain']})  "
          f"{row['formula']}  charge {row['formal_charge']}  {row['n_heavy']} heavy atoms")
    print(f"  SMILES          : {row['smiles']}   (from {row.get('smiles_source')})")
    if row.get("stereo_note"):
        print(f"  ! stereochemistry: {row['stereo_note']}")
        if not row.get("stereo_ok", True):
            print(f"    scored as    : {row.get('smiles_as_built')}")
    print(f"  Vina score_only : {row['vina_score_only']:+.3f} kcal/mol   (pose as predicted)")
    print(f"  Vina local_min  : {row['vina_local_min']:+.3f} kcal/mol   "
          f"(relaxed in place, moved {row['local_min_rmsd']:.2f} A)")
    if row.get("dock_best") is not None:
        print(f"  Vina redock     : {row['dock_best']:+.3f} kcal/mol   "
              f"(best pose, {row['dock_best_rmsd']:.2f} A from the predicted pose)")
        if row.get("dock_best_within_2A") is not None:
            print(f"    best pose within 2 A of prediction: {row['dock_best_within_2A']:+.3f} kcal/mol")
    if row.get("af3_ranking_score") is not None:
        print(f"  AlphaFold3      : ranking_score {row['af3_ranking_score']}  "
              f"ptm {row.get('af3_ptm')}  iptm {row.get('af3_iptm')}")
    if row.get("vina_score_only", 0) > 0:
        print("  ! positive score_only: the predicted pose clashes; trust local_min instead")
    print(f"  files           : {row['workdir']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="CIF file(s) and/or directories to search for *.cif")
    ap.add_argument("-o", "--out", help="write a CSV here (default: only for >1 structure)")
    ap.add_argument("-w", "--workdir", default="vina_work",
                    help="per-structure working dirs (default vina_work/)")
    ap.add_argument("--smiles", help="ligand SMILES (overrides the one in the CIF / input file)")
    ap.add_argument("--ligand_chain", help="ligand chain ID (default: every non-polymer chain)")
    ap.add_argument("--ph", type=float, default=7.4,
                    help="receptor protonation pH for Open Babel (default 7.4)")
    ap.add_argument("--dock", action="store_true",
                    help="also redock in a box around the predicted pose")
    ap.add_argument("--padding", type=float, default=6.0,
                    help="dock box padding around ligand per side, A (default 6)")
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--num_modes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="structures processed in parallel")
    ap.add_argument("--cpu", type=int, default=0,
                    help="CPUs per Vina run (default: all if -j 1, else 1)")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if cached")
    ap.add_argument("--keep_files", action="store_true", help="also keep receptor.pdb")
    args = ap.parse_args()

    for exe in (VINA, OBABEL):
        if not shutil.which(exe):
            sys.exit(f"'{exe}' not found on PATH")

    cifs = find_cifs(args.inputs)
    if not cifs:
        sys.exit(f"no .cif files found under {args.inputs}")
    if args.cpu == 0:
        args.cpu = os.cpu_count() if args.jobs == 1 else 1

    a = vars(args).copy()
    a["_args_key"] = json.dumps({k: a[k] for k in
                                 ("smiles", "ligand_chain", "ph", "dock", "padding",
                                  "exhaustiveness", "num_modes", "seed")}, sort_keys=True)

    jobs = [{"cif": c, "workdir": os.path.abspath(os.path.join(args.workdir, workdir_tag(c))),
             "args": a} for c in cifs]

    if len(jobs) > 1:
        print(f"{len(jobs)} structure(s); dock={'yes' if args.dock else 'no'}; "
              f"jobs={args.jobs}; cpu/vina={args.cpu}", flush=True)

    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = [r for rs in ex.map(process, jobs) for r in rs]
    else:
        rows = [r for j in jobs for r in process(j)]

    for r in rows:
        report(r)

    out = args.out or (os.path.join(args.workdir, "vina_scores.csv") if len(rows) > 1 else None)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["name"]))
        print(f"\nwrote {out}")

    nok = sum(r.get("status") == "ok" for r in rows)
    nskip = sum(r.get("status") == "skipped" for r in rows)
    nfail = len(rows) - nok - nskip
    if len(rows) > 1:
        print(f"{nok} ok, {nskip} skipped, {nfail} failed"
              + (f"; details in {args.workdir}/*/error.log" if nfail else ""))
    # A job script must be able to tell: any failure is a non-zero exit, skips are not.
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
