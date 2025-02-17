#python3 final_proba_model.py 1bdm Caf.mol2 "CN1C=NC2=C1C(=O)N(N(C(=O)N2C)C" lys
#into 2 
import sys
import json

def convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def run_pipeline(pdb_id, mol2_path, smiles, ligand_code):

    import subprocess
    import pandas as pd
    import os
    from rdkit import Chem
    from padelpy import from_smiles
    import csv
    import shutil
    import re
    from Bio.PDB import PDBParser
    import pandas as pd

    # Convert variables to strings
    pdb_id = str(pdb_id)
    ligand_code = str(ligand_code)
    smiles = str(smiles)
    mol2_path = str(mol2_path)

    # Define global variables
    pdb_file = pdb_id
    ligand_code = ligand_code
    SMILES =smiles
    mol2_file = mol2_path
    current_directory = '/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio'
    csv_name=pdb_id
    output_directory = os.path.join(current_directory, f"{csv_name}_folder")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    ##DPOCKET##
    def generate_input_txt(pdb_id, ligand_id):
        # Generate the input text file with file extensions
        input_txt = f'{pdb_id}.{ligand_id}_input.txt'
        with open(input_txt, 'w') as file:
            file.write(f'/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/{pdb_id} {ligand_id}')

        return input_txt

    def run_dpocket(pdb_id, ligand_id,output_directory):
        # Include file extensions in pdb_id
        pdb_id_with_extension = f'{pdb_id}.pdb'
        print(pdb_id_with_extension)
        # Generate input text file
        input_txt = generate_input_txt(pdb_id_with_extension, ligand_id)

        # Run dpocket command
        dpocket_command = f"dpocket -f {input_txt} -o {pdb_id}"
        subprocess.run(dpocket_command, shell=True)

        # Define the file names with file extensions
        txt_file = f'{csv_name}_exp.txt'
        csv_file = f'{csv_name}_data.csv'

        # Read the text file using pandas, skipping the first row
        df = pd.read_csv(txt_file, sep='\s+', header=None)

        # Reset the index before writing to the CSV file
        df.reset_index(drop=True, inplace=True)

        # Specify csv_path here
        csv_path = os.path.join(output_directory, csv_file)

        # Write the data to a CSV file in the specified current folder, excluding the index column
        df.to_csv(csv_path, index=False, header=True)

        # Now, remove the first row from the CSV file
        with open(csv_path, 'r') as file:
            lines = file.readlines()

        # Rewrite the CSV file excluding the first row
        with open(csv_path, 'w') as file:
            file.writelines(lines[1:])

    # Call the run_dpocket function with global variables
    run_dpocket(pdb_file, ligand_code,output_directory)

    ##PaDel##

    output_csv = os.path.join(output_directory, f'{csv_name}_descriptors.csv')
    output_csv1 = os.path.join(output_directory, f'{csv_name}_cavity_interactions.csv')
    def generate_descriptors(smiles_str):
        nonlocal output_csv
        # Set batch size and number of batches
        batch_size = 1  # Since you have a single SMILES string
        num_batches = 1

        # Process the list in batches
        results = []
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = min(start_index + batch_size, 1)  # Only one SMILES string
            batch_list = [smiles_str]

            print(f"Processing batch {batch + 1}/{num_batches} with {len(batch_list)} SMILES IDs")

            for i, smiles_id in enumerate(batch_list):
                print(f"Calculating descriptors for SMILES ID {start_index + i + 1}")
                try:
                    descriptor = from_smiles([smiles_id], timeout=600)

                    # Create a dictionary for the SMILES ID and its descriptors
                    result = {"SMILES ID": smiles_id}
                    result.update(descriptor[0])  # Assuming descriptor is a dictionary

                    results.append(result)
                except Exception as e:
                    print(f"Error processing SMILES ID {smiles_id}: {str(e)}")
                    continue

            print("Batch processed.\n")

        # Convert results to a DataFrame
        df = pd.DataFrame(results)

        # Save the descriptors CSV file in the specified output directory
        df.to_csv(output_csv, index=False)
        print(f"Descriptors CSV file saved at: {output_csv}")

    # Example usage with global variable SMILES
    smiles_str = SMILES  # Assuming SMILES is a valid SMILES string
    generate_descriptors(smiles_str)
    ##NACCESS##
    def run_naccess():
        nonlocal pdb_file, current_directory

        # Include the file extension
        pdb_file_with_extension = f"{pdb_file}.pdb"

        # Run NACCESS on the provided PDB file and capture the output
        result = subprocess.run(['/home/iiitd/softwares/naccess', pdb_file_with_extension], cwd=current_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Check if there was an error
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
        

        # Define the path to the generated NACCESS output file (.rsa)
        naccess_output_file = os.path.join(current_directory, f"{csv_name}.rsa")

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
        nonlocal output_directory

        with open(csv_output_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([ "PDB","all_atoms_rsa", "total_side_rsa", "main_chain_rsa", "non_polar_rsa", "all_polar_rsa"])
            
            for data in data_list:
                csv_writer.writerow(data)

    def process_pdb_file():
        nonlocal pdb_file, current_directory

        # Run NACCESS on the PDB file and get the output .rsa file
        naccess_output_file = run_naccess()

        # Extract RSA values from the .rsa file
        rsa_data = extract_rsa_values(naccess_output_file)

        # Define the path for the CSV output file
        csv_output_file = os.path.join(output_directory, f'{csv_name}_rsa_value.csv')

        # Write the RSA values to a CSV file
        write_to_csv(csv_output_file, rsa_data)

        return naccess_output_file, csv_output_file

    # Example usage with global variables
    naccess_output_file, csv_output_file = process_pdb_file()

    print(f"NACCESS output: {naccess_output_file}")
    print(f"RSA values saved to: {csv_output_file}")

    ##DSSP ON WHOLE PROTEIN##
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
                        break  # Increment count once per line for each keyword set

        return keyword_counts

    def save_to_csv(all_keyword_counts, output_file, total_residues):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['file name'] + list(all_keyword_counts[list(all_keyword_counts.keys())[0]].keys())
            writer.writerow(header)

            for file_name, counts in all_keyword_counts.items():
                row = [file_name]
                for count in counts.values():
                    row.append(count)
                writer.writerow(row)

    input_pdb_file = os.path.join(current_directory, f"{pdb_file}.pdb")
    dssp_output_file = os.path.join(output_directory, f"{pdb_file}.dssp")
    output_csv_file = os.path.join(output_directory, f"{csv_name}_dssp.csv")

    run_dssp(input_pdb_file, dssp_output_file)

    keyword_sets = {
            'perc_Helix': ['H  >', 'H  <', 'H  3>', 'H  3<', 'H  4', 'H  X'],
            'perc_BetaSheet': ['B'],
            'H-bond': ['S  >', 'S  <', 'S  3', 'S  3<'],
            'perc_Bend': ['T']
        }

    all_keyword_counts = {}

    lines = parse_dssp_file(dssp_output_file)
    keyword_counts = search_and_count_keywords(lines, keyword_sets)
    total_residues = sum(keyword_counts.values())  # Calculate total residues

    all_keyword_counts[os.path.basename(input_pdb_file)] = keyword_counts

    save_to_csv(all_keyword_counts, output_csv_file, total_residues)

    print("Output saved to", output_csv_file)

    ##LIGBUILDER##
    ##LIGBUILDER##
    # Specify the full paths to your PDB and MOL2 files
    pdb_file = os.path.join(current_directory, f"{pdb_file}.pdb")
    mol2_file = os.path.join(current_directory, f"{ligand_code}.mol2")
    output_folder = output_directory
    input_template = os.path.join(current_directory, f"cavity.input")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the original input template content
    with open(input_template, 'r') as template_file:
        template_content = template_file.read()

    # Replace the placeholders with actual paths without any prefix
    updated_content = template_content.replace("receptor/1db4.pdb", pdb_file).replace("receptor/1db4.mol2", mol2_file)

    # Create a folder for the current PDB-ligand pair
    pair_output_folder = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{os.path.splitext(os.path.basename(mol2_file))[0]}")
    print(pair_output_folder)
    os.makedirs(pair_output_folder, exist_ok=True)

    # Save the updated content to the cavity.input file
    cavity_input_path = os.path.join(pair_output_folder, "cavity.input")
    print(cavity_input_path)
    with open(cavity_input_path, 'w') as cavity_input_file:
        cavity_input_file.write(updated_content)

    # Change directory and run the cavity command using the generated input file
    # Set ulimit within the script
       
    import subprocess
   # Run the shell script
    os.system(f"bash /10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/run_cavity.sh {cavity_input_path}")
       
   # Move the output files to the pair-specific output folder
    for file_name in os.listdir("."):
        if file_name.startswith("output") and not os.path.isdir(file_name):
            print(file_name)
            shutil.move(file_name, os.path.join(pair_output_folder, file_name))


    # Specify the directory containing .pdb files
    pdb_directory = current_directory
        
    # Specify the directory to copy the selected surface .pdb files
    output_directory1 = os.path.join(output_directory, f"selected_surface")

    # Specify the directory to copy the corresponding 'cavity' .pdb files
    cavity_directory = os.path.join(output_directory, f"selected_cavity")


    # Create the output directory if it doesn't exist
    os.makedirs(output_directory1, exist_ok=True)

    # Create the cavity directory if it doesn't exist
    os.makedirs(cavity_directory, exist_ok=True)

    # Initialize variables to store the maximum drug score and corresponding surface/cavity files
    max_drug_score = -float('inf')
    selected_surface_file = None
    selected_cavity_file = None

    # Regular expression pattern to match surface and cavity file names
    file_pattern = re.compile(r'(.*)_surface_(\d+)\.pdb')
    cavity_pattern = re.compile(r'(.*)_cavity_(\d+)\.pdb')

    # Iterate through .pdb files in the directory
    input_pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
    print(input_pdb_base)
    print(current_directory)    
    for pdb_file in os.listdir(current_directory):
        if pdb_file.endswith('.pdb') and pdb_file.startswith(input_pdb_base):
     
            pdb_id = None
            file_number = None
            pdb_file_path = os.path.join(pdb_directory, pdb_file)
		
            drug_score = None

            # Try to match the file name with the regular expression patterns
            surface_match = file_pattern.match(pdb_file)
            cavity_match = cavity_pattern.match(pdb_file)

            if surface_match:
                pdb_id = surface_match.group(1)
                file_number = int(surface_match.group(2))

            # Open and read the .pdb file
            with open(pdb_file_path, 'r') as pdb_file_content:
                for line in pdb_file_content:
                    if line.startswith('REMARK   6 DrugScore :'):
                        # Extract the DrugScore value
                        drug_score = float(line.split(':')[1].strip())
                        break

            # Check if a DrugScore was found in the .pdb file
            if pdb_id and drug_score is not None:
                # Update the maximum DrugScore and corresponding surface/cavity files
                if drug_score > max_drug_score:
                    print("This one")
                    print(pdb_file_path)
                    max_drug_score = drug_score
                    selected_surface_file = pdb_file_path
                    selected_cavity_file = os.path.join(pdb_directory, f'{pdb_id}_cavity_{file_number}.pdb')
                    print(selected_cavity_file)


    # Process the selected surface file with the maximum drug score
    if selected_surface_file:
        # Initialize variables to store the extracted values
        drug_score = None
        average_pkd = None
        total_surface_area = None

        # Open and read the selected surface .pdb file
        with open(selected_surface_file, 'r') as selected_pdb_file:
            for line in selected_pdb_file:
                if line.startswith('REMARK   6 DrugScore :'):
                    # Extract the DrugScore value
                    drug_score = float(line.split(':')[1].strip())
                elif line.startswith('REMARK   5 Predict Average pKd'):
                    # Extract the Pkd value without additional information in parentheses
                    pkd_info = line.split(':')[1].strip()
                    pkd_match = re.search(r'(\d+\.\d+) \(', pkd_info)
                    if pkd_match:
                        average_pkd = float(pkd_match.group(1))
                elif line.startswith('REMARK   5 Predict Average pKd:'):
                    # Extract the Pkd value without additional information in parentheses
                    pkd_info = line.split(':')[1].strip()
                    pkd_match = re.search(r'(\d+\.\d+) \(', pkd_info)
                    if pkd_match:
                        average_pkd = float(pkd_match.group(1))
                elif line.startswith('REMARK   4 Total Surface area is'):
                    # Extract the Total Surface area value
                    total_surface_area = float(line.split()[-2])

        # Print the extracted values
        print(f"DrugScore: {drug_score}")
        print(f"Predict Average pKd: {average_pkd}")
        print(f"Total Surface area: {total_surface_area}")

        # Save information to a CSV file
        csv_output_file = os.path.join(output_directory, f'{csv_name}_surface_info.csv')

        with open(csv_output_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write header row
            writer.writerow(['PDB ID', 'DrugScore', 'Predicted Average pKd', 'Total Surface Area (A^2)'])

            # Write data row
            writer.writerow([pdb_id, drug_score, average_pkd, total_surface_area])

        print(f"Results saved to: {csv_output_file}")

        # Copy the selected surface file to the output directory
        shutil.copy(selected_surface_file, output_directory1)

        # Copy the corresponding 'cavity' file to the cavity directory
        shutil.copy(selected_cavity_file, cavity_directory)

        print(f"Selected surface .pdb file copied to: {output_directory1}")
        print(f"Corresponding 'cavity' .pdb file copied to: {cavity_directory}")
    else:
        print("No surface file selected.")


    #print(f"Processed {os.path.splitext(os.path.basename(pdb_file))[0]} - {os.path.splitext(os.path.basename(mol2_file))[0]}")
    print(f"Processed {input_pdb_base} - {os.path.splitext(os.path.basename(mol2_file))[0]}")
    print("All iterations are complete.")

    #NACCESS OF CAVITY
    def run_naccess2():
        
        # Include the file extension
        pdb_file_with_extension_cavity = os.path.basename(selected_cavity_file)
        print(pdb_file_with_extension_cavity)

        # Run NACCESS on the provided PDB file and capture the output
        subprocess.check_output(['/home/iiitd/softwares/naccess', pdb_file_with_extension_cavity], cwd=current_directory, universal_newlines=True)

        # Define the path to the generated NACCESS output file (.rsa)
        naccess_output_file2 = os.path.join(current_directory, f"{os.path.splitext(os.path.basename(selected_cavity_file))[0]}.rsa")
        print(naccess_output_file2)

        return naccess_output_file2

    def extract_rsa_values(naccess_output_file2):
        data_list2 = []
        

        with open(naccess_output_file2, 'r') as file:
            for line in file:
                if line.startswith("TOTAL"):
                    total_values = line.strip().split()
                    data_list2.append(total_values)

        return data_list2 
        print(data_list2)

    def write_to_csv(csv_output_file2, data_list2):
        with open(csv_output_file2, 'w', newline='') as csv_file2:
            csv_writer = csv.writer(csv_file2)
            csv_writer.writerow(["PDB","all_atoms_rsa_cavity", "total_side_rsa_cavity", "main_chain_rsa_cavity", "non_polar_rsa_cavity", "all_polar_rsa_cavity"])

            for data in data_list2:
                csv_writer.writerow(data)
    def process_pdb_file():
        

        # Run NACCESS on the PDB file and get the output .rsa file
        naccess_output_file2 = run_naccess2()

        # Extract RSA values from the .rsa file
        rsa_data2 = extract_rsa_values(naccess_output_file2)

        # Define the path for the CSV output file
        csv_output_file2 = os.path.join(output_directory, f'{csv_name}_cavity_rsa_value.csv')

        # Write the RSA values to a CSV file
        write_to_csv(csv_output_file2, rsa_data2)

        return naccess_output_file2, csv_output_file2

    # Example usage with global variables

    naccess_output_file2, csv_output_file2 = process_pdb_file()

    print(f"NACCESS2 output: {naccess_output_file2}")
    print(f"RSA2 values saved to: {csv_output_file2}")

    ##BIOPYTHON##
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

    def count_aromatic_dimers(structure, stacking_distance_threshold=5.0):
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
                                        distance = calculate_distance(atom1, atom2)
                                        if distance <= stacking_distance_threshold:
                                            dimer_count += 1
        return dimer_count

    def process_cavity(selected_cavity_file, s_no_start=1):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", selected_cavity_file)

        percentage_aromatic_residues = 0
        average_bfactor = 0
        aromatic_dimers = 0

        total_residues = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':
                        total_residues += 1
                        if is_aromatic(residue):
                            percentage_aromatic_residues += 1

        percentage_aromatic = (percentage_aromatic_residues / total_residues) * 100
        average_bfactor = calculate_average_bfactor(structure)
        aromatic_dimers = count_aromatic_dimers(structure)

        # Add s.no column
        results = [s_no_start, percentage_aromatic_residues, average_bfactor, aromatic_dimers]

        return results

    s_no_start = 1  # You can adjust the starting serial number
    results = process_cavity(selected_cavity_file, s_no_start)
    df = pd.DataFrame([results], columns=["s.no", "perc_aromatic_residues", "Average B-Factor", "Pi-Pi Stacking Count"])
    df.to_csv(output_csv1, index=False)


    selected_cavity_file = selected_cavity_file # Replace with the path to your selected cavity file
    current_directory = current_directory  # Replace with the path to your current directory
    csv_name = csv_name # Specify your desired CSV name
    output_csv1 = os.path.join(output_directory, f'{csv_name}_cavity_interactions.csv')  # Specify the output CSV file name


    ##H-BOND##
    import csv
    from Bio import PDB
    import numpy as np
    import os

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

        if hydrogen_bond_distances:
            average_hydrogen_bond_distance = np.mean(hydrogen_bond_distances)
        else:
            average_hydrogen_bond_distance = 'N/A'

        return average_hydrogen_bond_distance, hydrogen_bond_count

    def process_single_pdb(pdb_file_path, ligand_code, output_csv):
        try:
            avg_hbond_distance, num_hbonds = analyze_interactions(pdb_file_path, ligand_code)
            with open(output_csv, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['PDB File', 'Average H-Bond Distance', 'Number of H-Bonds'])
                csv_writer.writerow([os.path.basename(pdb_file_path), avg_hbond_distance, num_hbonds])
            print(f"Results saved to {output_csv}")
        except FileNotFoundError:
            print(f"File not found: {pdb_file_path}")
        except Exception as e:
            print(f"Error processing {pdb_file_path}: {str(e)}")

    # Set global variables
    pdb_file_path = os.path.join(current_directory, f"{csv_name}.pdb")
    ligand_code = ligand_code
    output_csv2 = os.path.join(output_directory, f'{csv_name}_H-BOND_interactions.csv')
    # Call the function with global variables as arguments
    process_single_pdb(pdb_file_path, ligand_code, output_csv2)


    ##SECONDARY STRUCTURES OF CAVITY##
    # Define a function to calculate percentages of helices and sheets
    def calculate_percentages(pdb_file_path):
        # Define variables to store helix and sheet counts
        helix_count = 0
        sheet_count = 0
        total_residue_count = 0

        # Open and read the PDB file
        with open(pdb_file_path, 'r') as pdb_file:
            for line in pdb_file:
                if line.startswith("HELIX"):
                    helix_count += 1
                elif line.startswith("SHEET"):
                    sheet_count += 1
                elif line.startswith("ATOM"):
                    # Assuming an "ATOM" line represents a residue
                    total_residue_count += 1

        # Calculate percentages
        percentage_helix = (helix_count / total_residue_count) * 100 if total_residue_count > 0 else 0
        percentage_sheet = (sheet_count / total_residue_count) * 100 if total_residue_count > 0 else 0

        return percentage_helix, percentage_sheet

    # Specify the folder containing cavity PDB files
    pdb_folder_path = cavity_directory # Create a list to store results
    results = []

    # Iterate over PDB files in the folder
    for filename in os.listdir(pdb_folder_path):
        if filename.endswith(".pdb"):
            pdb_file_path = os.path.join(pdb_folder_path, filename)
            percentages = calculate_percentages(pdb_file_path)
            results.append(( percentages[0], percentages[1]))

    # Define the output CSV file path
    output_csv_path = os.path.join(output_directory, f'{csv_name}_SS_perc.csv')
    # Write the results to a CSV file
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['perc_helix_cavity', 'perc_sheet_cavity'])
        csv_writer.writerows(results)

    print(f"Results saved to {output_csv_path}")

    ##COMBINE CSV###

    import os
    import pandas as pd

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Flag to check if it's the first file being processed
    first_file = True

    # Get the output directory
    csv_directory =  output_directory # Replace with your actual output directory
    print (csv_directory)
    # Iterate through all CSV files in the output directory
    for csv_file in os.listdir(csv_directory):
        if csv_file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            csv_path = os.path.join(csv_directory, csv_file)
            try:
                df = pd.read_csv(csv_path)

                # Exclude the first column (if it's not the first file)
                if not first_file:
                    df = df.iloc[:, 1:]

                # Append the data to the combined DataFrame
                combined_data = pd.concat([combined_data, df], axis=1)

                # Update the flag after processing the first file
                first_file = False

            except Exception as e:
                print(f"Error reading CSV file {csv_path}: {e}")

    # Specify the output CSV file path for the combined data
    output_combined_csv = os.path.join(output_directory, 'combined_data_all_files.csv')

    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv(output_combined_csv, index=False)

    print(f"Combined data from all CSV files in '{output_directory}' saved to {output_combined_csv}")




    ##MODEL LOADING##

    import pandas as pd

    Trained_features=os.path.join(current_directory,'trained_features.txt')
    # Load the list of features from the text file
    with open(Trained_features, 'r') as file:
        feature_list = [line.strip() for line in file]

    # Load the combined CSV file
    combined_csv = pd.read_csv(output_combined_csv)
    print("CSV File Loaded Successfully")

    # Select only the columns corresponding to the features in the list
    selected_data = combined_csv[feature_list]
    print(selected_data)

    # Specify the output CSV file path for the selected features
    output_selected_csv = os.path.join(output_directory,'selected_data.csv')

    # Save the new CSV file with selected features
    selected_data.to_csv(output_selected_csv, index=False)
    print(f"CSV File with selected features saved to {output_selected_csv}")

    import joblib
    import pandas as pd
    import xgboost as xgb

# Load the trained XGBoost model from the .pkl file
    model_filename_pkl = os.path.join(current_directory, 'xgboost_proba_model.pkl')
    loaded_model_pkl = joblib.load(model_filename_pkl)
    print("Trained XGBoost model loaded successfully from .pkl file")

# Assuming you have a single entry CSV file named 'single_entry.csv'
    single_entry_filename = os.path.join(output_directory, output_selected_csv)
# Load the single entry data from the CSV file
    single_entry_data = pd.read_csv(single_entry_filename)

    if not hasattr(loaded_model_pkl, '_is_fitted') or not loaded_model_pkl._is_fitted():
    # If the model is not fitted, you need to fit it before making predictions
    # Assuming you have training data X_train and labels y_train
    # loaded_model_pkl.fit(X_train, y_train)
    	print("Model was not fitted. You should fit the model before making predictions.")

# Make predictions using the loaded XGBoost model
    predictions_pkl = loaded_model_pkl.predict(single_entry_data)

# Calculate the probability scores for both classes (0 = Antagonist, 1 = Agonist)
    probabilities_pkl = loaded_model_pkl.predict_proba(single_entry_data)
    # Calculate the confidence score based on agreement among trees
    # Extract raw prediction contributions from individual trees
    # Use the Booster object to get leaf outputs for individual trees
    tree_predictions = loaded_model_pkl.get_booster().predict(
        xgb.DMatrix(single_entry_data), pred_leaf=True
    )
    # Agreement among trees
    tree_class_predictions = (tree_predictions >= 0.5).astype(int)
    confidence_scores = np.mean(tree_class_predictions == np.expand_dims(predictions_pkl, axis=1), axis=1)


# Print the prediction and probabilities for each complex
    for i in range(len(predictions_pkl)):
        prediction = "Agonist" if predictions_pkl[i] == 1 else "Antagonist"
        probability_agonist = float(probabilities_pkl[i][1])  # Probability of being Agonist (class 1)
        probability_antagonist = float(probabilities_pkl[i][0])  # Probability of being Antagonist (class 0)
        confidence_score = confidence_scores[i]  # Confidence score from agreement among trees
        
        print(f"Prediction for complex {i+1}: {prediction}")
        print(f"Probability score of the complex for being Agonist : {probability_agonist:.4f}")
        print(f"Probability score of the complex for being Antagonist : {probability_antagonist:.4f}")
        print(f"Confidence score (based on agreement among trees): {confidence_score:.4f}")

    # Optionally, you can return or store these results as needed
        json_output = {
            "Prediction": prediction,
            "Probability Score for Agonist": probability_agonist,
            "Probability Score for Antagonist": probability_antagonist,
            "Confidence Score (Agreement)": confidence_score
         }
        print(json.dumps(json_output, indent=4))

# Display or use the predictions as needed
    #result_dict = {"predictions": predictions_pkl.tolist()}
    #return result_dict


if __name__=="__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python your_script.py <pdb_id> <mol2_path> <smiles> <ligand_code>")
        sys.exit(1)

    # Assign command-line arguments to variables
    pdb_id, mol2_path, smiles, ligand_code = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # Call the run_pipeline function with the obtained values
    run_pipeline(pdb_id, mol2_path, smiles, ligand_code)
    #print(json.dumps(predictions_pkl))
