#!/usr/bin/env python

# ==============================================================================
# GOOGLE COLAB SETUP:
# ==============================================================================
# Before running this script in a Google Colab notebook,
# execute the following commands in a Colab cell to install dependencies:
#
# !apt-get update

#

# !apt-get install -y openbabel autodock-vina wget
#!pip install "numpy<3.0"
#!pip install biopython rdkit-pypi mdanalysis MDAnalysisTests

# ==============================================================================
import argparse
import os
import subprocess
import sys
import shutil
import re

# Try to import necessary libraries and provide guidance if they are missing.
try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB import Structure as BioPDBStructure # Alias to avoid conflict
    from Bio.PDB import Model as BioPDBModel
    from Bio.PDB import Chain as BioPDBChain
except ImportError:
    print("Error: Biopython not found. Please ensure it's installed (e.g., pip install biopython).")
    print("In Colab, run: !pip install biopython after ensuring numpy<2.0 is installed.")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit not found. Please ensure it's installed (e.g., pip install rdkit-pypi).")
    print("In Colab, run: !pip install rdkit-pypi after ensuring numpy<2.0 is installed.")
    sys.exit(1)

try:
    import MDAnalysis as mda
    from MDAnalysis.exceptions import NoDataError
except ImportError:
    print("Error: MDAnalysis not found. Please ensure it's installed (e.g., pip install mdanalysis MDAnalysisTests).")
    print("In Colab, run: !pip install mdanalysis MDAnalysisTests after ensuring numpy<2.0 is installed.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy not found. Please ensure it's installed (e.g., pip install \"numpy<2.0\").")
    print("In Colab, run: !pip install \"numpy<2.0\"")
    sys.exit(1)


# --- Configuration ---
# Path to executables (should be in PATH after Colab installation)
OBABEL_EXECUTABLE = "obabel"
VINA_EXECUTABLE = "vina"

# Default names for intermediate and output files
PROTEIN_PDB = "protein.pdb"
LIGAND_PDB = "ligand.pdb"
LIGAND_H_PDB = "ligand_h.pdb" # Ligand with hydrogens from RDKit
LIGAND_PDBQT = "ligand.pdbqt"

PROTEIN_H_PDB = "protein_h.pdb" # Protein with hydrogens from OpenBabel
PROTEIN_PDBQT = "protein.pdbqt"

VINA_CONFIG_FILE = "vina_config.txt"
VINA_OUT_PDBQT = "docked_ligand.pdbqt"
VINA_LOG_FILE = "vina_run.log" # Vina output will be redirected here

# --- Helper Functions ---

def check_executables():
    """Checks if OpenBabel and Vina are accessible."""
    if not shutil.which(OBABEL_EXECUTABLE):
        print(f"Error: OpenBabel ('{OBABEL_EXECUTABLE}') not found in PATH.")
        print("Please ensure it's installed. In Colab, run the setup cells at the top of the script.")
        sys.exit(1)
    if not shutil.which(VINA_EXECUTABLE):
        print(f"Error: AutoDock Vina ('{VINA_EXECUTABLE}') not found in PATH.")
        print("Please ensure it's installed. In Colab, run the setup cells at the top of the script.")
        sys.exit(1)
    print("Found OpenBabel and Vina executables in PATH.")

class NonHetSelect(Select):
    """Selects non-heteroatoms (protein)."""
    def accept_residue(self, residue):
        return residue.id[0] == ' '

class HetSelect(Select):
    """Selects heteroatoms (ligand/cofactors), excluding water."""
    def __init__(self, ligand_resname=None):
        self.ligand_resname = ligand_resname

    def accept_residue(self, residue):
        is_het = residue.id[0].startswith('H_')
        is_water = residue.get_resname().strip() in ['HOH', 'WAT']
        
        if self.ligand_resname:
            return is_het and not is_water and residue.get_resname().strip() == self.ligand_resname
        else:
            return is_het and not is_water

def parse_cif_to_pdb(cif_filepath, protein_out_pdb, ligand_out_pdb, target_ligand_resname=None):
    """
    Parses a CIF file, separates protein and a specified ligand (or first HETATM),
    and saves them as PDB files.
    """
    print(f"Parsing CIF file: {cif_filepath}")
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", cif_filepath)
    except Exception as e:
        print(f"Error parsing CIF file {cif_filepath}: {e}")
        sys.exit(1)

    io = PDBIO()
    io.set_structure(structure)

    print(f"Saving protein to {protein_out_pdb}")
    io.save(protein_out_pdb, NonHetSelect())

    found_ligand_residue = None
    # Prioritize specified ligand name
    if target_ligand_resname:
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0].startswith('H_') and \
                       residue.get_resname().strip() == target_ligand_resname and \
                       residue.get_resname().strip() not in ['HOH', 'WAT']:
                        found_ligand_residue = residue
                        break
                if found_ligand_residue: break
            if found_ligand_residue: break
        
        if not found_ligand_residue:
            print(f"Warning: Specified ligand residue name '{target_ligand_resname}' not found as a non-water HETATM.")
            print("Attempting to find any non-water HETATM as ligand...")
    
    # If no specific ligand or specified not found, find first non-water HETATM
    if not found_ligand_residue:
        for model in structure:
            for chain in model:
                for residue in chain:
                    if HetSelect().accept_residue(residue): # Generic check for non-water HETATM
                        found_ligand_residue = residue
                        print(f"Found generic HETATM: {residue.get_resname().strip()} in chain {chain.id}, residue ID {residue.id}")
                        break
                if found_ligand_residue: break
            if found_ligand_residue: break

    if found_ligand_residue:
        original_full_resname = found_ligand_residue.get_resname().strip()
        
        pdb_compatible_resname = original_full_resname.upper()
        if len(original_full_resname) > 3:
            pdb_compatible_resname = original_full_resname[:3].upper() 
            print(f"Warning: Original ligand residue name '{original_full_resname}' ({len(original_full_resname)} chars) is longer than 3 characters. "
                  f"Using truncated name '{pdb_compatible_resname}' for the intermediate PDB file for RDKit compatibility.")
        
        print(f"Saving ligand ({original_full_resname}) to {ligand_out_pdb} using PDB resname '{pdb_compatible_resname}' for internal PDB.")
        
        lig_struct = BioPDBStructure.Structure("ligand_only")
        lig_model = BioPDBModel.Model(0)
        lig_chain = BioPDBChain.Chain('L') 
        
        hetflag = 'H_' + pdb_compatible_resname 
        new_residue_id = (hetflag, found_ligand_residue.id[1], found_ligand_residue.id[2])

        copied_residue = Residue(new_residue_id, pdb_compatible_resname, found_ligand_residue.segid)
        for atom_obj in found_ligand_residue: 
            copied_atom = Atom(atom_obj.name, atom_obj.coord, atom_obj.bfactor, atom_obj.occupancy, atom_obj.altloc,
                               atom_obj.fullname, atom_obj.serial_number, element=atom_obj.element)
            copied_residue.add(copied_atom)

        lig_chain.add(copied_residue)
        lig_model.add(lig_chain)
        lig_struct.add(lig_model)
        
        tmp_io = PDBIO()
        tmp_io.set_structure(lig_struct)
        tmp_io.save(ligand_out_pdb)
        return ligand_out_pdb, original_full_resname 
    else:
        print("Error: No suitable ligand (non-water HETATM) found in the CIF file.")
        print("If your ligand is a standard residue or part of a polymer, this script may need modification.")
        sys.exit(1)


def prepare_ligand_pdbqt(ligand_pdb_in, ligand_pdbqt_out):
    """
    Prepares ligand PDBQT:
    1. Reads PDB, adds hydrogens with RDKit, saves as ligand_h.pdb.
    2. Converts ligand_h.pdb to PDBQT using OpenBabel.
    """
    print(f"Preparing ligand: {ligand_pdb_in} -> {ligand_pdbqt_out}")
    try:
        mol = Chem.MolFromPDBFile(ligand_pdb_in, removeHs=False, sanitize=False)
        if mol is None:
            print(f"Error: RDKit could not read ligand PDB: {ligand_pdb_in}")
            if os.path.exists(ligand_pdb_in):
                print(f"Contents of problematic {ligand_pdb_in} (first 10 lines):")
                try:
                    with open(ligand_pdb_in, 'r') as f_err:
                        for i, line in enumerate(f_err):
                            if i < 10:
                                print(line.strip())
                            else:
                                break
                except Exception as e_read:
                    print(f"Could not read {ligand_pdb_in}: {e_read}")
            else:
                print(f"{ligand_pdb_in} does not exist.")
            sys.exit(1)
        
        mol = Chem.RemoveHs(mol) 
        mol_h = Chem.AddHs(mol, addCoords=True) 
        if mol_h is None:
            print("Error: RDKit failed to add hydrogens to ligand.")
            sys.exit(1)
        
        Chem.MolToPDBFile(mol_h, LIGAND_H_PDB)
        print(f"Ligand with RDKit hydrogens saved to: {LIGAND_H_PDB}")

    except Exception as e:
        print(f"Error during RDKit processing of ligand '{ligand_pdb_in}': {e}")
        sys.exit(1)

    try:
        # Removed -xr flag for ligand preparation. 
        # OpenBabel should correctly identify rotatable bonds and assign atom types for a flexible ligand.
        cmd = [OBABEL_EXECUTABLE, LIGAND_H_PDB, "-O", ligand_pdbqt_out, "--partialcharge", "gasteiger"]
        print(f"Running OpenBabel for ligand: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            check=False, 
            capture_output=True, text=True
        )
        if process.returncode != 0:
            print(f"Error running OpenBabel for ligand PDBQT conversion (Exit code: {process.returncode}):")
            print(f"Command: {' '.join(cmd)}")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
            sys.exit(1)
        if not os.path.exists(ligand_pdbqt_out) or os.path.getsize(ligand_pdbqt_out) == 0:
            print(f"Error: OpenBabel ligand PDBQT conversion failed to produce output file or file is empty: {ligand_pdbqt_out}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
            sys.exit(1)
        print(f"Ligand PDBQT saved to: {ligand_pdbqt_out}")
    except Exception as e:
        print(f"An unexpected error occurred during OpenBabel ligand PDBQT conversion: {e}")
        sys.exit(1)


def prepare_protein_pdbqt(protein_pdb_in, protein_pdbqt_out, ph=7.4):
    """
    Prepares protein PDBQT:
    1. Adds hydrogens using OpenBabel at specified pH, saves as protein_h.pdb.
    2. Converts protein_h.pdb to PDBQT using OpenBabel (as receptor).
    """
    print(f"Preparing protein: {protein_pdb_in} -> {protein_pdbqt_out}")

    try:
        cmd_add_h = [OBABEL_EXECUTABLE, protein_pdb_in, "-O", PROTEIN_H_PDB, f"-p{ph}"]
        print(f"Running OpenBabel to add hydrogens to protein: {' '.join(cmd_add_h)}")
        process_add_h = subprocess.run(cmd_add_h, check=False, capture_output=True, text=True)
        if process_add_h.returncode != 0:
            print(f"Error running OpenBabel for adding hydrogens to protein (Exit code: {process_add_h.returncode}):")
            print(f"Command: {' '.join(cmd_add_h)}")
            print(f"Stdout: {process_add_h.stdout}")
            print(f"Stderr: {process_add_h.stderr}")
            sys.exit(1)
        if not os.path.exists(PROTEIN_H_PDB) or os.path.getsize(PROTEIN_H_PDB) == 0:
            print(f"Error: OpenBabel protein hydrogenation failed to produce output file or file is empty: {PROTEIN_H_PDB}")
            print(f"Command: {' '.join(cmd_add_h)}")
            print(f"Stdout: {process_add_h.stdout}")
            print(f"Stderr: {process_add_h.stderr}")
            sys.exit(1)
        print(f"Protein with OpenBabel hydrogens (pH {ph}) saved to: {PROTEIN_H_PDB}")
    except Exception as e:
        print(f"An unexpected error occurred during OpenBabel protein hydrogenation: {e}")
        sys.exit(1)

    try:
        # -xr flag is appropriate here for preparing a rigid receptor
        cmd_to_pdbqt = [OBABEL_EXECUTABLE, PROTEIN_H_PDB, "-O", protein_pdbqt_out, "-xr"]
        print(f"Running OpenBabel to convert protein to PDBQT: {' '.join(cmd_to_pdbqt)}")
        process_to_pdbqt = subprocess.run(cmd_to_pdbqt, check=False, capture_output=True, text=True)
        if process_to_pdbqt.returncode != 0:
            print(f"Error running OpenBabel for protein PDBQT conversion (Exit code: {process_to_pdbqt.returncode}):")
            print(f"Command: {' '.join(cmd_to_pdbqt)}")
            print(f"Stdout: {process_to_pdbqt.stdout}")
            print(f"Stderr: {process_to_pdbqt.stderr}")
            sys.exit(1)
        if not os.path.exists(protein_pdbqt_out) or os.path.getsize(protein_pdbqt_out) == 0:
            print(f"Error: OpenBabel protein PDBQT conversion failed to produce output file or file is empty: {protein_pdbqt_out}")
            print(f"Command: {' '.join(cmd_to_pdbqt)}")
            print(f"Stdout: {process_to_pdbqt.stdout}")
            print(f"Stderr: {process_to_pdbqt.stderr}")
            sys.exit(1)
        print(f"Protein PDBQT saved to: {protein_pdbqt_out}")
    except Exception as e:
        print(f"An unexpected error occurred during OpenBabel protein PDBQT conversion: {e}")
        sys.exit(1)


def get_vina_config_params(ligand_pdb_for_box, padding=10.0):
    """
    Calculates Vina box parameters (center, size) based on ligand coordinates.
    """
    print(f"Calculating Vina box parameters from: {ligand_pdb_for_box}")
    try:
        u = mda.Universe(ligand_pdb_for_box)
    except (NoDataError, FileNotFoundError, Exception) as e:
        print(f"Error: MDAnalysis could not load ligand PDB for box calculation: {ligand_pdb_for_box}")
        print(f"Details: {e}")
        if os.path.exists(ligand_pdb_for_box) and os.path.getsize(ligand_pdb_for_box) == 0:
             print("The ligand PDB file for box calculation is empty.")
        sys.exit(1)
    
    if not u.atoms: 
        print(f"Error: No atoms found in {ligand_pdb_for_box} by MDAnalysis. Cannot define docking box.")
        sys.exit(1)
        
    lig_atoms = u.select_atoms("all") 
    if not lig_atoms: 
        print(f"Error: No atoms selected in {ligand_pdb_for_box} by MDAnalysis. Check PDB content.")
        sys.exit(1)

    center = lig_atoms.center_of_geometry()
    positions = lig_atoms.positions
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    
    for i in range(3):
        if max_coords[i] == min_coords[i]:
            max_coords[i] += 0.1 

    size = (max_coords - min_coords) + (2 * padding) 
    size = np.maximum(size, np.array([1.0, 1.0, 1.0])) 

    print(f"  Ligand Center (x,y,z): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")
    print(f"  Box Size (x,y,z): {size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f} (includes {padding}A padding on each side)")
    return center, size


def write_vina_config(receptor_pdbqt, ligand_pdbqt, out_pdbqt, 
                      center_xyz, size_xyz, config_filepath,
                      num_modes=10, energy_range=3, exhaustiveness=8, seed=None):
    """Writes the Vina configuration file.
    Note: 'log' parameter is removed as it's not supported by Vina 1.2.3 config.
    Log will be handled by capturing stdout from subprocess.
    """
    print(f"Writing Vina configuration to: {config_filepath}")
    config_content = f"""
receptor = {os.path.abspath(receptor_pdbqt)}
ligand = {os.path.abspath(ligand_pdbqt)}

out = {os.path.abspath(out_pdbqt)}

center_x = {center_xyz[0]:.4f}
center_y = {center_xyz[1]:.4f}
center_z = {center_xyz[2]:.4f}

size_x = {size_xyz[0]:.4f}
size_y = {size_xyz[1]:.4f}
size_z = {size_xyz[2]:.4f}

num_modes = {num_modes}
energy_range = {energy_range}
exhaustiveness = {exhaustiveness}
"""
    if seed is not None:
        config_content += f"seed = {seed}\n"

    with open(config_filepath, 'w') as f:
        f.write(config_content)
    print("Vina configuration file written.")


def run_vina(vina_executable, config_filepath, log_filepath):
    """
    Runs AutoDock Vina and captures its stdout to a log file.
    """
    print(f"Running AutoDock Vina with config: {config_filepath}")
    print(f"Vina output (log) will be saved to: {log_filepath}")
    command = [vina_executable, "--config", os.path.abspath(config_filepath)]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=os.getcwd())
        
        # Write Vina's stdout to the log file
        with open(log_filepath, 'w') as log_f:
            log_f.write("Vina STDOUT:\n")
            log_f.write(process.stdout)
            if process.stderr:
                log_f.write("\nVina STDERR:\n")
                log_f.write(process.stderr)
        
        print("Vina run completed successfully.")
        print(f"Full Vina output saved to {log_filepath}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running AutoDock Vina (Exit code: {e.returncode}):")
        print(f"Command: {' '.join(e.cmd)}")
        with open(log_filepath, 'w') as log_f:
            log_f.write(f"Error running Vina. Command: {' '.join(e.cmd)}\n")
            log_f.write("Vina STDOUT:\n")
            log_f.write(e.stdout)
            log_f.write("\nVina STDERR:\n")
            log_f.write(e.stderr)
        print(f"Vina output (error) saved to {log_filepath}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Vina executable '{vina_executable}' not found. Is it in your PATH?")
        print("In Colab, ensure the setup cells at the top of the script were run correctly.")
        sys.exit(1)

def extract_binding_affinity(vina_out_pdbqt_filepath, num_modes_to_extract=1):
    """
    Extracts binding affinity for the top mode(s) from Vina output PDBQT.
    """
    print(f"Extracting binding affinity from: {vina_out_pdbqt_filepath}")
    affinities = []
    try:
        with open(vina_out_pdbqt_filepath, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        affinity = float(parts[3]) 
                        affinities.append(affinity)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse affinity from Vina output line: {line.strip()} - {e}")
                if len(affinities) >= num_modes_to_extract: 
                    break 
        
        if affinities:
            print(f"Top binding affinity (kcal/mol): {affinities[0]}")
            if len(affinities) > 1:
                print(f"Affinities for top {len(affinities)} modes: {affinities}")
            return affinities[0] 
        else:
            print(f"Warning: No binding affinity lines (REMARK VINA RESULT) found in Vina output file: {vina_out_pdbqt_filepath}")
            print("Check the Vina log file for errors or details. The docking might have failed or produced no valid poses.")
            return None

    except FileNotFoundError:
        print(f"Error: Vina output PDBQT file not found: {vina_out_pdbqt_filepath}")
        return None

def cleanup_files(files_to_delete):
    """Deletes specified intermediate files."""
    print("\nCleaning up intermediate files...")
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"  Removed: {f_path}")
            except OSError as e:
                print(f"  Warning: Could not remove {f_path} - {e}")
        else:
            print(f"  Skipped (not found): {f_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated docking pipeline using Biopython, RDKit, MDAnalysis, and AutoDock Vina. Designed for Colab compatibility.",
        formatter_class=argparse.RawTextHelpFormatter, 
        epilog="""
Example usage:
  python script_name.py input.cif --ligand_resname LIG --padding 10.0

In Google Colab:
1. Upload this script and your CIF file.
2. Run the setup commands in a Colab cell (see top of script).
3. Run the script from a Colab cell:
   !python script_name.py your_complex.cif --ligand_resname XYZ
"""
    )
    parser.add_argument("cif_file", help="Input CIF file containing protein-ligand complex.")
    parser.add_argument("--ligand_resname", help="Residue name of the ligand in the CIF file (e.g., 'LIG', 'STI'). If not provided, the script will try to identify the first non-water HETATM.", default=None)
    parser.add_argument("--padding", type=float, default=10.0, help="Padding around ligand for Vina box size (Angstroms). Default: 10.0 A on each side.")
    parser.add_argument("--num_modes", type=int, default=10, help="Number of binding modes for Vina. Default: 10.")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Exhaustiveness of Vina search. Default: 8.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Vina. Default: None (Vina uses time-based seed).")
    parser.add_argument("--protein_ph", type=float, default=7.4, help="pH for protonating the protein using OpenBabel. Default: 7.4.")
    parser.add_argument("--keep_files", action="store_true", help="Keep intermediate files instead of deleting them.")
    
    args = parser.parse_args()

    print("--- Starting Docking Pipeline ---")
    print(f"Input CIF: {args.cif_file}")
    if args.ligand_resname:
        print(f"Target Ligand Residue Name: {args.ligand_resname}")
    print(f"Vina Box Padding: {args.padding} A")


    if not os.path.exists(args.cif_file):
        print(f"Error: Input CIF file not found: {args.cif_file}")
        sys.exit(1)

    check_executables() 

    ligand_pdb_for_box, actual_ligand_name = parse_cif_to_pdb(args.cif_file, PROTEIN_PDB, LIGAND_PDB, args.ligand_resname)
    if not ligand_pdb_for_box or not actual_ligand_name :
        print("Exiting due to error in CIF parsing or ligand identification.")
        sys.exit(1)
    print(f"Successfully parsed CIF. Identified ligand: {actual_ligand_name}")

    prepare_ligand_pdbqt(LIGAND_PDB, LIGAND_PDBQT)
    prepare_protein_pdbqt(PROTEIN_PDB, PROTEIN_PDBQT, ph=args.protein_ph)

    center_xyz, size_xyz = get_vina_config_params(LIGAND_PDB, padding=args.padding) 

    write_vina_config(PROTEIN_PDBQT, LIGAND_PDBQT, VINA_OUT_PDBQT,
                      center_xyz, size_xyz, VINA_CONFIG_FILE,
                      num_modes=args.num_modes, exhaustiveness=args.exhaustiveness, seed=args.seed)

    run_vina(VINA_EXECUTABLE, VINA_CONFIG_FILE, VINA_LOG_FILE) 

    best_affinity = extract_binding_affinity(VINA_OUT_PDBQT, num_modes_to_extract=args.num_modes)

    if best_affinity is not None:
        print(f"\n--- Docking Workflow Completed ---")
        print(f"Best Binding Affinity for {actual_ligand_name} with {os.path.basename(args.cif_file)}: {best_affinity:.3f} kcal/mol")
        print(f"Output PDBQT with docked poses: {os.path.abspath(VINA_OUT_PDBQT)}")
        print(f"Vina log file: {os.path.abspath(VINA_LOG_FILE)}")
    else:
        print(f"\n--- Docking Workflow Completed with Issues ---")
        print("Could not determine binding affinity. Please check log files and Vina output PDBQT.")
        print(f"Attempted Vina output PDBQT: {os.path.abspath(VINA_OUT_PDBQT)}")
        print(f"Vina log file: {os.path.abspath(VINA_LOG_FILE)}")


    if not args.keep_files:
        files_to_remove = [
            PROTEIN_PDB, LIGAND_PDB, LIGAND_H_PDB, 
            PROTEIN_H_PDB, 
            VINA_CONFIG_FILE, 
        ]
        cleanup_files(files_to_remove)
    else:
        print("\nIntermediate files have been kept as requested.")
        print(f"  Protein PDB: {os.path.abspath(PROTEIN_PDB)}")
        print(f"  Ligand PDB: {os.path.abspath(LIGAND_PDB)}")
        print(f"  Ligand PDBQT: {os.path.abspath(LIGAND_PDBQT)}")
        print(f"  Protein PDBQT: {os.path.abspath(PROTEIN_PDBQT)}")
        print(f"  Vina Config: {os.path.abspath(VINA_CONFIG_FILE)}")

if __name__ == "__main__":
    main()

#!python docking.py 5op_model.cif --ligand_resname LIG --num_modes 5
