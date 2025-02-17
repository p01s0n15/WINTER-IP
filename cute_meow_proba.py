# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀
# ⠀⠀⠀⠀⢀⡴⣆⠀⠀⠀⠀⠀⣠⡀⠀⠀⠀⠀⠀⠀⣼⣿⡗⠀⠀⠀⠀
# ⠀⠀⠀⣠⠟⠀⠘⠷⠶⠶⠶⠾⠉⢳⡄⠀⠀⠀⠀⠀⣧⣿⠀⠀⠀⠀⠀
# ⠀⠀⣰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣤⣤⣤⣤⣤⣿⢿⣄⠀⠀⠀⠀
# ⠀⠀⡇⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠙⣷⡴⠶⣦
# ⠀⠀⢱⡀⠀⠉⠉⠀⠀⠀⠀⠛⠃⠀⢠⡟⠂⠀⠀⢀⣀⣠⣤⠿⠞⠛⠋
# ⣠⠾⠋⠙⣶⣤⣤⣤⣤⣤⣀⣠⣤⣾⣿⠴⠶⠚⠋⠉⠁⠀⠀⠀⠀⠀⠀
# ⠛⠒⠛⠉⠉⠀⠀⠀⣴⠟⣣⡴⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pandas as pd
import os
from rdkit import Chem
from padelpy import from_smiles
import csv
import shutil
import re
from Bio.PDB import PDBParser
import numpy as np
import joblib
import xgboost as xgb
from Bio import PDB

def convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

class PipelineTask:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.lock = threading.Lock()
        self.selected_cavity_file = None
        self.results = {}

    def run_dpocket(self, pdb_id, ligand_code, current_directory, csv_name):
        def generate_input_txt(pdb_id, ligand_id):
            input_txt = f'{pdb_id}.{ligand_id}_input.txt'
            with open(input_txt, 'w') as file:
                file.write(f'{current_directory}/{pdb_id}.pdb\t{ligand_id}')
            return input_txt

        input_txt = generate_input_txt(pdb_id, ligand_code)
        dpocket_command = f"dpocket -f {input_txt} -o {pdb_id}"
        subprocess.run(dpocket_command, shell=True, check=True)

        txt_file = f'{csv_name}_exp.txt'
        csv_file = f'{csv_name}_data.csv'
        
        df = pd.read_csv(txt_file, sep='\s+', header=None)
        df.reset_index(drop=True, inplace=True)
        
        csv_path = os.path.join(self.output_directory, csv_file)
        df.to_csv(csv_path, index=False, header=True)
        
        with open(csv_path, 'r') as file:
            lines = file.readlines()
        with open(csv_path, 'w') as file:
            file.writelines(lines[1:])
            
        self.results['dpocket'] = csv_path

    def generate_descriptors(self, smiles_str, csv_name):
        output_csv = os.path.join(self.output_directory, f'{csv_name}_descriptors.csv')
        
        with self.lock:
            try:
                descriptor = from_smiles([smiles_str], timeout=600)
                result = {"SMILES ID": smiles_str}
                result.update(descriptor[0])
                
                df = pd.DataFrame([result])
                df.to_csv(output_csv, index=False)
                self.results['descriptors'] = output_csv
                
            except Exception as e:
                print(f"Error generating descriptors: {str(e)}")

    def run_naccess(self, pdb_file, csv_name, current_directory):
        print(f"Running NACCESS on PDB file: {pdb_file}")
        def extract_rsa_values(naccess_output_file):
            data_list = []
            with open(naccess_output_file, 'r') as file:
                for line in file:
                    if line.startswith("TOTAL"):
                        total_values = line.strip().split()
                        data_list.append(total_values)
            return data_list

        with self.lock:
            try:
                subprocess.run(['/home/iiitd/softwares/naccess', pdb_file], 
                             cwd=current_directory, check=True)
                
                naccess_output_file = os.path.join(current_directory, f"{csv_name}.rsa")
                rsa_data = extract_rsa_values(naccess_output_file)
                
                csv_output_file = os.path.join(self.output_directory, f'{csv_name}_rsa_value.csv')
                
                with open(csv_output_file, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["PDB", "all_atoms_rsa", "total_side_rsa", 
                                       "main_chain_rsa", "non_polar_rsa", "all_polar_rsa"])
                    for data in rsa_data:
                        csv_writer.writerow(data)
                        
                self.results['naccess'] = csv_output_file
                
            except Exception as e:
                print(f"Error in NACCESS analysis: {str(e)}")

    def run_dssp(self, pdb_file, output_dssp, csv_name):
        def parse_dssp_file(file_path):
            with open(file_path, 'r') as file:
                return file.readlines()

        def search_and_count_keywords(lines, keyword_sets):
            keyword_counts = {key: 0 for key in keyword_sets.keys()}
            for line in lines:
                for keyword_set_name, keywords in keyword_sets.items():
                    for keyword in keywords:
                        if keyword.lower() in line.lower():
                            keyword_counts[keyword_set_name] += 1
                            break
            return keyword_counts

        with self.lock:
            try:
                subprocess.run(f"dssp -i {pdb_file} -o {output_dssp}", shell=True, check=True)
                
                keyword_sets = {
                    'perc_Helix': ['H  >', 'H  <', 'H  3>', 'H  3<', 'H  4', 'H  X'],
                    'perc_BetaSheet': ['B'],
                    'H-bond': ['S  >', 'S  <', 'S  3', 'S  3<'],
                    'perc_Bend': ['T']
                }
                
                lines = parse_dssp_file(output_dssp)
                keyword_counts = search_and_count_keywords(lines, keyword_sets)
                
                output_csv = os.path.join(self.output_directory, f'{csv_name}_dssp.csv')
                with open(output_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['file name'] + list(keyword_counts.keys()))
                    writer.writerow([os.path.basename(pdb_file)] + list(keyword_counts.values()))
                    
                self.results['dssp'] = output_csv
                
            except Exception as e:
                print(f"Error in DSSP analysis: {str(e)}")

    def run_ligbuilder(self, pdb_file, mol2_file, current_directory, ligand_code):
        with self.lock:
            try:
                # Create cavity input file
                input_template = os.path.join(current_directory, "cavity.input")
                with open(input_template, 'r') as template_file:
                    template_content = template_file.read()
                
                updated_content = template_content.replace("receptor/1db4.pdb", pdb_file).replace("receptor/1db4.mol2", mol2_file)
                
                pair_output_folder = os.path.join(self.output_directory, 
                                                f"{os.path.splitext(os.path.basename(pdb_file))[0]}_"
                                                f"{os.path.splitext(os.path.basename(mol2_file))[0]}")
                os.makedirs(pair_output_folder, exist_ok=True)
                
                cavity_input_path = os.path.join(pair_output_folder, "cavity.input")
                with open(cavity_input_path, 'w') as cavity_input_file:
                    cavity_input_file.write(updated_content)
                
                # Run cavity analysis
                os.system(f"bash /10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/run_cavity.sh {cavity_input_path}")
                
                # Process results
                for file_name in os.listdir("."):
                    if file_name.startswith("output") and not os.path.isdir(file_name):
                        shutil.move(file_name, os.path.join(pair_output_folder, file_name))
                
                self.process_cavity_results(pdb_file, current_directory)
                
            except Exception as e:
                print(f"Error in LigBuilder analysis: {str(e)}")

    def process_cavity_results(self, pdb_file, current_directory):
        try:
            # Find cavity with maximum drug score
            max_drug_score = -float('inf')
            input_pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
            
            for pdb_file in os.listdir(current_directory):
                if pdb_file.endswith('.pdb') and pdb_file.startswith(input_pdb_base):
                    pdb_file_path = os.path.join(current_directory, pdb_file)
                    
                    with open(pdb_file_path, 'r') as pdb_file_content:
                        for line in pdb_file_content:
                            if line.startswith('REMARK   6 DrugScore :'):
                                drug_score = float(line.split(':')[1].strip())
                                if drug_score > max_drug_score:
                                    max_drug_score = drug_score
                                    self.selected_cavity_file = pdb_file_path
                                break
            
            # Process selected cavity
            if self.selected_cavity_file:
                self.analyze_selected_cavity()
                
        except Exception as e:
            print(f"Error processing cavity results: {str(e)}")

    def analyze_selected_cavity(self):
        try:
            # Extract cavity information
            drug_score = None
            average_pkd = None
            total_surface_area = None
            
            with open(self.selected_cavity_file, 'r') as selected_pdb_file:
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
            
            # Save results
            csv_output_file = os.path.join(self.output_directory, f'{os.path.splitext(os.path.basename(self.selected_cavity_file))[0]}_surface_info.csv')
            with open(csv_output_file, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['PDB ID', 'DrugScore', 'Predicted Average pKd', 'Total Surface Area (A^2)'])
                writer.writerow([os.path.splitext(os.path.basename(self.selected_cavity_file))[0], 
                               drug_score, average_pkd, total_surface_area])
            
            self.results['cavity_analysis'] = csv_output_file
            
        except Exception as e:
            print(f"Error analyzing selected cavity: {str(e)}")

    def analyze_cavity_interactions(self):
        if not self.selected_cavity_file:
            return
            
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", self.selected_cavity_file)
            
            def is_aromatic(residue):
                return residue.get_resname() in ["TRP", "TYR", "PHE", "HIS"]
            
            def calculate_distance(atom1, atom2):
                return atom1 - atom2
            
            def calculate_average_bfactor(structure):
                b_factors = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                b_factors.append(atom.get_bfactor())
                return sum(b_factors) / len(b_factors) if b_factors else 0
            
            def count_aromatic_dimers(structure, threshold=5.0):
                dimer_count = 0
                for model in structure:
                    for chain in model:
                        residues = list(chain)
                        for i in range(len(residues) - 1):
                            if is_aromatic(residues[i]):
                                for j in range(i + 1, len(residues)):
                                    if is_aromatic(residues[j]):
                                        for atom1 in residues[i]:
                                            for atom2 in residues[j]:
                                                if calculate_distance(atom1, atom2) <= threshold:
                                                    dimer_count += 1
                return dimer_count
            
            total_residues = 0
            aromatic_residues = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':
                            total_residues += 1
                            if is_aromatic(residue):
                                aromatic_residues += 1
            
            percentage_aromatic = (aromatic_residues / total_residues * 100) if total_residues > 0 else 0
            average_bfactor = calculate_average_bfactor(structure)
            aromatic_dimers = count_aromatic_dimers(structure)
            
            results = [1, percentage_aromatic, average_bfactor, aromatic_dimers]
            df = pd.DataFrame([results], columns=["s.no", "perc_aromatic_residues", 
                                                "Average B-Factor", "Pi-Pi Stacking Count"])
            
            output_csv = os.path.join(self.output_directory, 
                                    f'{os.path.splitext(os.path.basename(self.selected_cavity_file))[0]}_cavity_interactions.csv')
            df.to_csv(output_csv, index=False)
            
            self.results['interactions'] = output_csv
            
        except Exception as e:
            print(f"Error analyzing cavity interactions: {str(e)}")

def run_pipeline(pdb_id, mol2_path, smiles, ligand_code):
    # Convert variables to strings
    pdb_id = str(pdb_id)
    ligand_code = str(ligand_code)
    smiles = str(smiles)
    mol2_path = str(mol2_path)

    # Define directories
    current_directory = '/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio'
    csv_name = pdb_id
    output_directory = os.path.join(current_directory, f"{csv_name}_folder")
    os.makedirs(output_directory, exist_ok=True)

    # Initialize pipeline tasks
    pipeline = PipelineTask(output_directory)

    # Create thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        # Submit dpocket task
        futures.append(
            executor.submit(
                pipeline.run_dpocket,
                pdb_id,
                ligand_code,
                current_directory,
                csv_name
            )
        )

        # Submit PaDEL descriptor generation
        futures.append(
            executor.submit(
                pipeline.generate_descriptors,
                smiles,
                csv_name
            )
        )

        # Submit NACCESS analysis
        futures.append(
            executor.submit(
                pipeline.run_naccess,
                f"{pdb_id}.pdb",
                csv_name,
                current_directory
            )
        )

        # Submit DSSP analysis
        futures.append(
            executor.submit(
                pipeline.run_dssp,
                os.path.join(current_directory, f"{pdb_id}.pdb"),
                os.path.join(output_directory, f"{pdb_id}.dssp"),
                csv_name
            )
        )

        # Wait for all initial tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel task: {str(e)}")

        # Run LigBuilder analysis (sequential due to dependencies)
        pipeline.run_ligbuilder(
            os.path.join(current_directory, f"{pdb_id}.pdb"),
            os.path.join(current_directory, f"{ligand_code}.mol2"),
            current_directory,
            ligand_code
        )

        # Analyze cavity interactions
        pipeline.analyze_cavity_interactions()

    # Combine results and run model predictions
    try:
        combine_results_and_predict(pipeline.results, output_directory, current_directory)
    except Exception as e:
        print(f"Error in final analysis: {str(e)}")

def combine_results_and_predict(results, output_directory, current_directory):
    """Combines all CSV files and runs the prediction model."""
    
    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()
    first_file = True

    # Combine all CSV results
    for result_type, csv_file in results.items():
        try:
            df = pd.read_csv(csv_file)
            print(f"Processing {csv_file}:")  # Debug
            print(df.columns)  # Debug
            if not first_file:
                df = df.iloc[:, 1:]
            combined_data = pd.concat([combined_data, df], axis=1)
            first_file = False
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")

    # Save combined data
    output_combined_csv = os.path.join(output_directory, 'combined_data_all_files.csv')
    combined_data.to_csv(output_combined_csv, index=False)
    print("Combined data columns:")  # Debug
    print(combined_data.columns)  # Debug

    # Load trained features
    trained_features_path = os.path.join(current_directory, 'trained_features.txt')
    with open(trained_features_path, 'r') as file:
        feature_list = [line.strip() for line in file]
    print("Trained features:")  # Debug
    print(feature_list)  # Debug

    # Select relevant features
    try:
        selected_data = combined_data[feature_list]
    except KeyError as e:
        print(f"Missing features in combined data: {e}")
        return

    output_selected_csv = os.path.join(output_directory, 'selected_data.csv')
    selected_data.to_csv(output_selected_csv, index=False)

    # Load and run the model
    model_path = os.path.join(current_directory, 'xgboost_proba_model.pkl')
    loaded_model = joblib.load(model_path)

    # Make predictions
    predictions = loaded_model.predict(selected_data)
    probabilities = loaded_model.predict_proba(selected_data)

    # Calculate confidence scores
    tree_predictions = loaded_model.get_booster().predict(
        xgb.DMatrix(selected_data),
        pred_leaf=True
    )
    tree_class_predictions = (tree_predictions >= 0.5).astype(int)
    confidence_scores = np.mean(tree_class_predictions == np.expand_dims(predictions, axis=1), axis=1)

    # Format and return results
    for i in range(len(predictions)):
        prediction = "Agonist" if predictions[i] == 1 else "Antagonist"
        probability_agonist = float(probabilities[i][1])
        probability_antagonist = float(probabilities[i][0])
        confidence_score = confidence_scores[i]

        result = {
            "Prediction": prediction,
            "Probability Score for Agonist": probability_agonist,
            "Probability Score for Antagonist": probability_antagonist,
            "Confidence Score (Agreement)": float(confidence_score)
        }

        print(json.dumps(result, indent=4))
        return result

def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <pdb_id> <mol2_path> <smiles> <ligand_code>")
        sys.exit(1)

    pdb_id, mol2_path, smiles, ligand_code = sys.argv[1:5]

    try:
        result = run_pipeline(pdb_id, mol2_path, smiles, ligand_code)
        print(json.dumps(result, default=convert))
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()