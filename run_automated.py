import sys
import subprocess
import requests
from rdkit import Chem
from rdkit.Chem import MolToSmiles

def fetch_smiles_from_pubchem(ligand_code):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{ligand_code}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return None

def convert_mol2_to_smiles(mol2_file):
    try:
        with open(mol2_file, "r") as file:
            mol_block = file.read()
        mol = Chem.MolFromMol2Block(mol_block, sanitize=True, removeHs=True)
        return MolToSmiles(mol) if mol else None
    except Exception as e:
        print(f"Error converting MOL2 to SMILES: {e}")
        return None

def main(pdb_file, mol2_file):
    # Extract ligand code from MOL2 file
    ligand_code = mol2_file.split('.')[0]
    
    # Try fetching SMILES from PubChem first
    smiles = fetch_smiles_from_pubchem(ligand_code)
    if not smiles:
        print("SMILES not found in PubChem, converting from MOL2 file...")
        smiles = convert_mol2_to_smiles(mol2_file)
    
    if not ligand_code or not smiles:
        print("Error: Could not extract ligand code or SMILES string.")
        sys.exit(1)
    
    # Run final_proba_model.py with extracted information
    command = ["python3", "final_proba_model.py", pdb_file, mol2_file, smiles, ligand_code]
    subprocess.run(command)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python auto_generate_smiles.py <pdb_file> <mol2_file>")
        sys.exit(1)
    
    pdb_file, mol2_file = sys.argv[1], sys.argv[2]
    main(pdb_file, mol2_file)
