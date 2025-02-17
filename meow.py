import streamlit as st
import subprocess

st.title("Run MEOWTMBPH Script")

pdb_file = st.file_uploader("Upload PDB File", type=["pdb"])
mol2_file = st.file_uploader("Upload MOL2 File", type=["mol2"])

if st.button("Run Script"):
    if pdb_file and mol2_file:
        with open("temp.pdb", "wb") as f:
            f.write(pdb_file.getbuffer())
        with open("temp.mol2", "wb") as f:
            f.write(mol2_file.getbuffer())
        result = subprocess.run(
            ["python3", "MEOWTMBPH.py", "temp", "temp.mol2"],
            capture_output=True,
            text=True,
        )
        st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)
    else:
        st.error("Please upload both PDB and MOL2 files.")