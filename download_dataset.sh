#!/bin/bash
# download_dataset.sh (FASTA sequences only)

# Create input target directories for each genus
mkdir -p ncbi_dataset_Aeromonas ncbi_dataset_Citrobacter ncbi_dataset_Helicobacter ncbi_dataset_Streptococcus

echo "[+] Downloading genome sequences (FASTA) per genus from NCBI..."
datasets download genome taxon Aeromonas --include genome --assembly-level complete --released-before 2024-01-01 --filename ncbi_dataset_Aeromonas/dataset.zip
datasets download genome taxon Citrobacter --include genome --assembly-level complete --released-before 2024-01-01 --filename ncbi_dataset_Citrobacter/dataset.zip
datasets download genome taxon Helicobacter --include genome --assembly-level complete --released-before 2024-01-01 --filename ncbi_dataset_Helicobacter/dataset.zip
datasets download genome taxon Streptococcus --include genome --assembly-source refseq --assembly-level complete --released-before 2024-01-01 --filename ncbi_dataset_Streptococcus/dataset.zip

echo "[+] Unpacking archives..."
unzip -q -o ncbi_dataset_Aeromonas/dataset.zip -d ncbi_dataset_Aeromonas/
unzip -q -o ncbi_dataset_Citrobacter/dataset.zip -d ncbi_dataset_Citrobacter/
unzip -q -o ncbi_dataset_Helicobacter/dataset.zip -d ncbi_dataset_Helicobacter/
unzip -q -o ncbi_dataset_Streptococcus/dataset.zip -d ncbi_dataset_Streptococcus/

echo "[+] Download and extraction completed successfully."