# Genomic Classification via FCGR and Spiking Neural P Systems (SNPS)

An academic and reproducible machine learning pipeline for multi-class bacterial genomic classification. This project bypasses standard sliding-window $k$-mer extraction by using continuous **Frequency Chaos Game Representation (FCGR)** mapped directly into discretized $28 \times 28$ matrices, followed by feature extraction and classification using **Spiking Neural P Systems (SNPS)**.

---

## ?? Project Overview

- **Input Data**: Complete genomic sequences (FASTA) from 4 bacterial genera (*Aeromonas*, *Citrobacter*, *Helicobacter*, *Streptococcus*).
- **Representation**: Direct $28 \times 28$ FCGR matrix mapping ($k \approx 4-5$ informativeness equivalent).
- **Architecture**: 4-Layer Spiking Neural Network (SNPS) + Ensemble Quantized Classifier (SVM & Logistic Regression).
- **Dataset Partition**: 157 total samples (117 Training / 40 Testing).

---

## ?? Repository Structure

```text
.
+-- main.py                # Python entry point for dataset loading & SNPS execution
+-- produce_fcgr.R         # R script for parallel fasta-to-FCGR 28x28 generation
+-- download_dataset.sh    # Fast dataset acquisition script (NCBI CLI)
+-- download_all_dataset.sh# Complete dataset acquisition script (NCBI CLI)
+-- requirements.txt       # Environment dependencies
+-- sps/                   # Spiking Neural P System core module
    +-- config.py          # Network architecture & dataset split settings
    +-- network.py         # SNPS training, feature extraction, & ensemble pipeline
    +-- snp_system.py      # Core spiking neuron implementation & simulation loop
    +-- p_neuron.py        # Individual neuron mechanics
    +-- spike_utils.py     # Rule transformation & event utilities
    +-- handle_csv.py      # Output & metric logging
    +-- exact_csv.py       # Exact rule CSV export utilities