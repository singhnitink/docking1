#!/usr/bin/env python3
"""
Build an OpenFold3 query JSON from a protein sequence and one or more ligands.

  make_query.py --name gsh --fasta MNPIS... --smiles 'SC[C@@H](...)N' -o query.json
  make_query.py --name gsh --fasta prot.fa --smiles-file ligands.smi -o query.json

--fasta takes either the protein sequence itself or a path to a FASTA file
holding exactly one sequence.

ligands.smi is one SMILES per line, optionally "<smiles><whitespace><name>".
Each ligand gets its own chain, so one query can hold several.

The protein is chain A; ligands take B, C, D, ... in order.
"""
import argparse
import json
import os
import string
import sys

FASTA_EXT = (".fa", ".fasta", ".faa", ".fas", ".seq", ".txt")
AMINO = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def read_fasta(path):
    """One sequence. Several records would otherwise be silently joined into one
    chimeric chain and folded as if it were a real protein."""
    records, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = []
                records.append((line[1:].strip(), cur))
            else:
                if cur is None:
                    cur = []
                    records.append(("", cur))
                cur.append(line)
    seqs = [(h, "".join(s)) for h, s in records if "".join(s)]
    if not seqs:
        sys.exit(f"no sequence found in {path}")
    if len(seqs) > 1:
        heads = ", ".join(h or "(no header)" for h, _ in seqs[:5])
        sys.exit(f"{path} holds {len(seqs)} sequences ({heads}{', ...' if len(seqs) > 5 else ''}); "
                 f"this pipeline folds one protein chain - put the one you want in its own file")
    return seqs[0][1]


def read_smiles_file(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def check_smiles(smiles):
    """Reject an unparseable SMILES now, before a GPU run is spent on it."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return
    RDLogger.DisableLog("rdApp.*")
    if Chem.MolFromSmiles(smiles) is None:
        sys.exit(f"RDKit cannot parse this SMILES: {smiles}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="query name (also the output subdirectory)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fasta", help="the protein: either the sequence itself "
                                   "(one-letter codes) or a path to a FASTA file")
    g.add_argument("--sequence", help="alias for --fasta")
    ap.add_argument("--smiles", action="append", default=[],
                    help="ligand SMILES (repeatable)")
    ap.add_argument("--smiles-file", help="file with one ligand SMILES per line")
    ap.add_argument("-o", "--out", default="query.json", help="output JSON (default query.json)")
    args = ap.parse_args()

    # either form may carry the sequence itself or a path to a FASTA file
    given = args.fasta if args.fasta else args.sequence
    if os.path.isfile(given):
        seq = read_fasta(given)
    else:
        # a value that looks like a path must not be folded as a "sequence"
        if os.sep in given or given.lower().endswith(FASTA_EXT):
            sys.exit(f"--fasta {given}: no such file")
        seq = given
        print(f"--fasta: treating the value as a literal sequence "
              f"({len(''.join(given.split()))} residues)", file=sys.stderr)
    seq = "".join(seq.split()).upper()
    bad = set(seq) - AMINO
    if bad:
        sys.exit(f"sequence contains non-amino-acid characters: {sorted(bad)}")
    if len(seq) < 10:
        sys.exit(f"sequence is only {len(seq)} residues long - is this really a protein?")

    ligands = list(args.smiles)
    if args.smiles_file:
        ligands += read_smiles_file(args.smiles_file)
    if not ligands:
        sys.exit("no ligands given; use --smiles and/or --smiles-file")
    for smi in ligands:
        check_smiles(smi)

    chain_ids = [c for c in string.ascii_uppercase if c != "A"]
    if len(ligands) > len(chain_ids):
        sys.exit(f"too many ligands ({len(ligands)}) for single-letter chain IDs")

    chains = [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq}]
    for i, smi in enumerate(ligands):
        chains.append({"molecule_type": "ligand", "chain_ids": [chain_ids[i]], "smiles": smi})

    query = {"queries": {args.name: {"chains": chains}}}
    with open(args.out, "w") as fh:
        json.dump(query, fh, indent=2)
    print(f"wrote {args.out}: {len(seq)} residues, {len(ligands)} ligand(s) "
          f"on chain(s) {', '.join(chain_ids[:len(ligands)])}")


if __name__ == "__main__":
    main()
