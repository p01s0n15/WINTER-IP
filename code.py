import sys
import json
import os
import subprocess
import shutil
import re
import csv
import concurrent.futures
import pandas as pd
import numpy as np
from rdkit import Chem
from padelpy import from_smiles
from Bio.PDB import PDBParser
import joblib
import xgboost as xgb

def convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def run_dpocket(pdb_id, ligand_id, output_directory):
    pdb_id_with_extension = f'{pdb_id}.pdb'
    input_txt = f'{pdb_id}.{ligand_id}_input.txt'
    
    with open(input_txt, 'w') as file:
        file.write(f'/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/{pdb_id} {ligand_id}')
    dpocket_command = f"dpocket -f {input_txt} -o {pdb_id}"
    subprocess.run(dpocket_command, shell=True)
    
    txt_file = f'{pdb_id}_exp.txt'
    csv_file = f'{pdb_id}_data.csv'
    df = pd.read_csv(txt_file, sep='\s+', header=None)
    df.reset_index(drop=True, inplace=True)
    
    csv_path = os.path.join(output_directory, csv_file)
    df.to_csv(csv_path, index=False, header=True)
    
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    with open(csv_path, 'w') as file:
        file.writelines(lines[1:])

def generate_descriptors(smiles_str, output_csv):
    results = []
    try:
        descriptor = from_smiles([smiles_str], timeout=600)
        result = {"SMILES ID": smiles_str}
        result.update(descriptor[0])
        results.append(result)
    except Exception as e:
        print(f"Error processing SMILES ID {smiles_str}: {str(e)}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Descriptors CSV file saved at: {output_csv}")

def run_naccess(pdb_file, current_directory):
    pdb_file_with_extension = f"{pdb_file}.pdb"
    result = subprocess.run(['/home/iiitd/softwares/naccess', pdb_file_with_extension], cwd=current_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    
    naccess_output_file = os.path.join(current_directory, f"{pdb_file}.rsa")
    return naccess_output_file

def extract_rsa_values(naccess_output_file):
    data_list = []
    with open(naccess_output_file, 'r') as file:
        for line in file:
            if line.startswith("TOTAL"):
                total_values = line.strip().split()
                data_list.append(total_values)
    return data_list

def write_to_csv(csv_output_file, data_list):
    with open(csv_output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["PDB", "all_atoms_rsa", "total_side_rsa", "main_chain_rsa", "non_polar_rsa", "all_polar_rsa"])
        for data in data_list:
            csv_writer.writerow(data)

def process_pdb_file(pdb_file, current_directory, output_directory):
    naccess_output_file = run_naccess(pdb_file, current_directory)
    rsa_data = extract_rsa_values(naccess_output_file)
    csv_output_file = os.path.join(output_directory, f'{pdb_file}_rsa_value.csv')
    write_to_csv(csv_output_file, rsa_data)
    return naccess_output_file, csv_output_file

def run_dssp(input_pdb, output_dssp):
    dssp_command = f"dssp -i {input_pdb} -o {output_dssp}"
    subprocess.run(dssp_command, shell=True)

def parse_dssp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def search_and_count_keywords(lines, keyword_sets):
    keyword_counts = {key: 0 for key in keyword_sets.keys()}
    for line in lines:
        for keyword_set_name, keywords in keyword_sets.items():
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    keyword_counts[keyword_set_name] += 1
                    break
    return keyword_counts

def save_to_csv(all_keyword_counts, output_file, total_residues):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['file name'] + list(all_keyword_counts[list(all_keyword_counts.keys())[0]].keys())
        writer.writerow(header)
        for file_name, counts in all_keyword_counts.items():
            row = [file_name] + [count for count in counts.values()]
            writer.writerow(row)

def process_cavity(selected_cavity_file, s_no_start=1):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", selected_cavity_file)
    
    percentage_aromatic_residues = 0
    total_residues = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    total_residues += 1
                    if residue.get_resname() in ["TRP", "TYR", "PHE", "HIS"]:
                        percentage_aromatic_residues += 1
    
    percentage_aromatic = (percentage_aromatic_residues / total_residues) * 100
    average_bfactor = sum(atom.get_bfactor() for atom in structure.get_atoms()) / len(list(structure.get_atoms()))
    aromatic_dimers = count_aromatic_dimers(structure)
    
    results = [s_no_start, percentage_aromatic_residues, average_bfactor, aromatic_dimers]
    return results

def count_aromatic_dimers(structure, stacking_distance_threshold=5.0):
    dimer_count = 0
    for model in structure:
        for chain in model:
            residues = list(chain)
            for i in range(len(residues) - 1):
                if residues[i].get_resname() in ["TRP", "TYR", "PHE", "HIS"]:
                    for j in range(i + 1, len(residues)):
                        if residues[j].get_resname() in ["TRP", "TYR", "PHE", "HIS"]:
                            for atom1 in residues[i]:
                                for atom2 in residues[j]:
                                    distance = np.linalg.norm(atom1.coord - atom2.coord)
                                    if distance <= stacking_distance_threshold:
                                        dimer_count += 1
    return dimer_count

def analyze_interactions(pdb_file_path, ligand_code, cutoff_distance=6.0):
    parser = PDB.PDBParser()
    structure = parser.get_structure("protein_structure", pdb_file_path)
    ns = PDB.NeighborSearch(list(structure.get_atoms()))
    
    interacting_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_code:
                    for ligand_atom in residue:
                        nearby_atoms = ns.search(ligand_atom.coord, cutoff_distance)
                        interacting_atoms.extend(nearby_atoms)
    
    interacting_atoms = list(set(interacting_atoms))
    
    hydrogen_bond_distances = []
    hydrogen_bond_count = 0
    for atom in interacting_atoms:
        if (atom.element == "H" and atom.get_parent().get_resname() == ligand_code) or \
        ((atom.element == "O" or atom.element == "N" or atom.element == "H") and
            atom.get_parent().get_resname() != ligand_code):
            for other_atom in interacting_atoms:
                if other_atom != atom:
                    distance = np.linalg.norm(atom.coord - other_atom.coord)
                    if distance <= cutoff_distance:
                        hydrogen_bond_distances.append(distance)
                        hydrogen_bond_count += 1
    
    average_hydrogen_bond_distance = np.mean(hydrogen_bond_distances) if hydrogen_bond_distances else 'N/A'
    return average_hydrogen_bond_distance, hydrogen_bond_count

def process_single_pdb(pdb_file_path, ligand_code, output_csv):
    try:
        avg_hbond_distance, num_hbonds = analyze_interactions(pdb_file_path, ligand_code)
        with open(output_csv, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['PDB File', 'Average H-Bond Distance', 'Number of H-Bonds'])
            csv_writer.writerow([os.path.basename(pdb_file_path), avg_hbond_distance, num_hbonds])
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error processing {pdb_file_path}: {str(e)}")

def calculate_percentages(pdb_file_path):
    helix_count = 0
    sheet_count = 0
    total_residue_count = 0
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("HELIX"):
                helix_count += 1
            elif line.startswith("SHEET"):
                sheet_count += 1
            elif line.startswith("ATOM"):
                total_residue_count += 1
    
    percentage_helix = (helix_count / total_residue_count) * 100 if total_residue_count > 0 else 0
    percentage_sheet = (sheet_count / total_residue_count) * 100 if total_residue_count > 0 else 0
    return percentage_helix, percentage_sheet

def combine_csv_files(csv_directory, output_combined_csv):
    combined_data = pd.DataFrame()
    first_file = True
    
    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_directory, csv_file)
            try:
                df = pd.read_csv(csv_path)
                if not first_file:
                    df = df.iloc[:, 1:]
                combined_data = pd.concat([combined_data, df], axis=1)
                first_file = False
            except Exception as e:
                print(f"Error reading CSV file {csv_path}: {e}")
    
    combined_data.to_csv(output_combined_csv, index=False)
    print(f"Combined data from all CSV files in '{csv_directory}' saved to {output_combined_csv}")

def load_model_and_predict(model_filename_pkl, single_entry_filename):
    loaded_model_pkl = joblib.load(model_filename_pkl)
    single_entry_data = pd.read_csv(single_entry_filename)
    
    if not hasattr(loaded_model_pkl, '_is_fitted') or not loaded_model_pkl._is_fitted():
        print("Model was not fitted. You should fit the model before making predictions.")
    
    predictions_pkl = loaded_model_pkl.predict(single_entry_data)
    probabilities_pkl = loaded_model_pkl.predict_proba(single_entry_data)
    
    tree_predictions = loaded_model_pkl.get_booster().predict(xgb.DMatrix(single_entry_data), pred_leaf=True)
    tree_class_predictions = (tree_predictions >= 0.5).astype(int)
    confidence_scores = np.mean(tree_class_predictions == np.expand_dims(predictions_pkl, axis=1), axis=1)
    
    for i in range(len(predictions_pkl)):
        prediction = "Agonist" if predictions_pkl[i] == 1 else "Antagonist"
        probability_agonist = float(probabilities_pkl[i][1])
        probability_antagonist = float(probabilities_pkl[i][0])
        confidence_score = confidence_scores[i]
        
        print(f"Prediction for complex {i+1}: {prediction}")
        print(f"Probability score of the complex for being Agonist : {probability_agonist:.4f}")
        print(f"Probability score of the complex for being Antagonist : {probability_antagonist:.4f}")
        print(f"Confidence score (based on agreement among trees): {confidence_score:.4f}")
        
        json_output = {
            "Prediction": prediction,
            "Probability Score for Agonist": probability_agonist,
            "Probability Score for Antagonist": probability_antagonist,
            "Confidence Score (Agreement)": confidence_score
        }
        print(json.dumps(json_output, indent=4))

def run_pipeline(pdb_id, mol2_path, smiles, ligand_code):
    pdb_id = str(pdb_id)
    ligand_code = str(ligand_code)
    smiles = str(smiles)
    mol2_path = str(mol2_path)
    
    current_directory = '/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio'
    output_directory = os.path.join(current_directory, f"{pdb_id}_folder")
    os.makedirs(output_directory, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_dpocket, pdb_id, ligand_code, output_directory),
            executor.submit(generate_descriptors, smiles, os.path.join(output_directory, f'{pdb_id}_descriptors.csv')),
            executor.submit(process_pdb_file, pdb_id, current_directory, output_directory)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")
    
    input_pdb_file = os.path.join(current_directory, f"{pdb_id}.pdb")
    dssp_output_file = os.path.join(output_directory, f"{pdb_id}.dssp")
    output_csv_file = os.path.join(output_directory, f"{pdb_id}_dssp.csv")
    
    run_dssp(input_pdb_file, dssp_output_file)
    
    keyword_sets = {
        'perc_Helix': ['H  >', 'H  <', 'H  3>', 'H  3<', 'H  4', 'H  X'],
        'perc_BetaSheet': ['B'],
        'H-bond': ['S  >', 'S  <', 'S  3', 'S  3<'],
        'perc_Bend': ['T']
    }
    
    lines = parse_dssp_file(dssp_output_file)
    keyword_counts = search_and_count_keywords(lines, keyword_sets)
    total_residues = sum(keyword_counts.values())
    
    all_keyword_counts = {os.path.basename(input_pdb_file): keyword_counts}
    save_to_csv(all_keyword_counts, output_csv_file, total_residues)
    
    pdb_file = os.path.join(current_directory, f"{pdb_id}.pdb")
    mol2_file = os.path.join(current_directory, f"{ligand_code}.mol2")
    output_folder = output_directory
    input_template = os.path.join(current_directory, f"cavity.input")
    
    with open(input_template, 'r') as template_file:
        template_content = template_file.read()
    
    updated_content = template_content.replace("receptor/1db4.pdb", pdb_file).replace("receptor/1db4.mol2", mol2_file)
    
    pair_output_folder = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{os.path.splitext(os.path.basename(mol2_file))[0]}")
    os.makedirs(pair_output_folder, exist_ok=True)
    
    cavity_input_path = os.path.join(pair_output_folder, "cavity.input")
    with open(cavity_input_path, 'w') as cavity_input_file:
        cavity_input_file.write(updated_content)
    
    os.system(f"bash /10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/run_cavity.sh {cavity_input_path}")
    
    for file_name in os.listdir("."):
        if file_name.startswith("output") and not os.path.isdir(file_name):
            shutil.move(file_name, os.path.join(pair_output_folder, file_name))
    
    output_directory1 = os.path.join(output_directory, f"selected_surface")
    cavity_directory = os.path.join(output_directory, f"selected_cavity")
    os.makedirs(output_directory1, exist_ok=True)
    os.makedirs(cavity_directory, exist_ok=True)
    
    max_drug_score = -float('inf')
    selected_surface_file = None
    selected_cavity_file = None
    
    file_pattern = re.compile(r'(.*)_surface_(\d+)\.pdb')
    cavity_pattern = re.compile(r'(.*)_cavity_(\d+)\.pdb')
    
    input_pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
    for pdb_file in os.listdir(current_directory):
        if pdb_file.endswith('.pdb') and pdb_file.startswith(input_pdb_base):
            pdb_file_path = os.path.join(current_directory, pdb_file)
            drug_score = None
            
            with open(pdb_file_path, 'r') as pdb_file_content:
                for line in pdb_file_content:
                    if line.startswith('REMARK   6 DrugScore :'):
                        drug_score = float(line.split(':')[1].strip())
                        break
            
            if drug_score is not None and drug_score > max_drug_score:
                max_drug_score = drug_score
                selected_surface_file = pdb_file_path
                selected_cavity_file = os.path.join(current_directory, f'{input_pdb_base}_cavity_{file_pattern.match(pdb_file).group(2)}.pdb')
    
    if selected_surface_file:
        drug_score = None
        average_pkd = None
        total_surface_area = None
        
        with open(selected_surface_file, 'r') as selected_pdb_file:
            for line in selected_pdb_file:
                if line.startswith('REMARK   6 DrugScore :'):
                    drug_score = float(line.split(':')[1].strip())
                elif line.startswith('REMARK   5 Predict Average pKd'):
                    pkd_info = line.split(':')[1].strip()
                    pkd_match = re.search(r'(\d+\.\d+) \(', pkd_info)
                    if pkd_match:
                        average_pkd = float(pkd_match.group(1))
                elif line.startswith('REMARK   4 Total Surface area is'):
                    total_surface_area = float(line.split()[-2])
        
        csv_output_file = os.path.join(output_directory, f'{pdb_id}_surface_info.csv')
        with open(csv_output_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['PDB ID', 'DrugScore', 'Predicted Average pKd', 'Total Surface Area (A^2)'])
            writer.writerow([pdb_id, drug_score, average_pkd, total_surface_area])
        
        shutil.copy(selected_surface_file, output_directory1)
        shutil.copy(selected_cavity_file, cavity_directory)
    
    naccess_output_file2, csv_output_file2 = process_pdb_file(selected_cavity_file, current_directory, output_directory)
    
    results = process_cavity(selected_cavity_file)
    df = pd.DataFrame([results], columns=["s.no", "perc_aromatic_residues", "Average B-Factor", "Pi-Pi Stacking Count"])
    df.to_csv(os.path.join(output_directory, f'{pdb_id}_cavity_interactions.csv'), index=False)
    
    process_single_pdb(os.path.join(current_directory, f"{pdb_id}.pdb"), ligand_code, os.path.join(output_directory, f'{pdb_id}_H-BOND_interactions.csv'))
    
    results = []
    for filename in os.listdir(cavity_directory):
        if filename.endswith(".pdb"):
            pdb_file_path = os.path.join(cavity_directory, filename)
            percentages = calculate_percentages(pdb_file_path)
            results.append((percentages[0], percentages[1]))
    
    output_csv_path = os.path.join(output_directory, f'{pdb_id}_SS_perc.csv')
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['perc_helix_cavity', 'perc_sheet_cavity'])
        csv_writer.writerows(results)
    
    combine_csv_files(output_directory, os.path.join(output_directory, 'combined_data_all_files.csv'))
    
    model_filename_pkl = os.path.join(current_directory, 'xgboost_proba_model.pkl')
    single_entry_filename = os.path.join(output_directory, 'selected_data.csv')
    load_model_and_predict(model_filename_pkl, single_entry_filename)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python final_proba_model.py <pdb_id> <mol2_path> <smiles> <ligand_code>")
        sys.exit(1)
    
    pdb_id, mol2_path, smiles, ligand_code = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    run_pipeline(pdb_id, mol2_path, smiles, ligand_code)