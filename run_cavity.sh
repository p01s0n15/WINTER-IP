#!/bin/bash

# Define paths
INPUT_FILE="/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/1bdm_folder/1bdm_Lys/cavity.input"
OUTPUT_DIR="/10tb-storage/akshatw/dhinkachika/dhinkachika/ip_bio/1bdm_folder/1bdm_Lys"
CAVITY_TOOL="/usr/local/bin/dpocket"  # Update this with the actual executable path

# Run cavity detection
$CAVITY_TOOL -i $INPUT_FILE -o $OUTPUT_DIR/cavity_output.txt

echo "Cavity detection completed."

