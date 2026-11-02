# Genomic Classification via FCGR and Spiking Neural P Systems (SNPS)

A SNPS simulator for bacterial genomic classification. This project bypasses standard sliding-window k-mer extraction by using continuous Frequency Chaos Game Representation (FCGR) mapped directly into discretized 28x28 matrices, followed by feature extraction and classification using SNPS.

## Project Overview

* **Input Data:** Complete genomic sequences (FASTA) from 4 bacterial genera (Aeromonas, Citrobacter, Helicobacter, Streptococcus).
* **Representation:** Direct 28x28 FCGR matrix mapping.
* **Architecture:** 4-Layer Spiking Neural P System with an ensemble classifier (SVM & Logistic Regression).
* **Dataset Partition:** An example with 157 total samples (117 Training / 40 Testing).

## Environment Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install R Dependencies

Run R in your terminal or environment and execute:

```r
install.packages(c("kaos", "seqinr", "foreach", "doParallel"))
```

## Complete Execution Pipeline

Execute the following commands in sequence to run the complete data acquisition, matrix generation, and model training workflow:

### Step 1: Download Genomic Dataset

```bash
bash download_dataset.sh
```

### Step 2: Extract 28x28 FCGR Matrices

```bash
Rscript produce_fcgr.R
```

### Step 3: Run Training and Classification Pipeline

```bash
python main.py -i dataset_28x28
```

## Configuration Settings

Dataset partition and matrix resolution parameters configured in `sps/config.py`:

* **Matrix Resolution:** 28x28 (`IMG_SHAPE = 28`)
* **Target Classes:** 4 (Aeromonas, Citrobacter, Helicobacter, Streptococcus)
* **Dataset Split:** 157 total samples (117 Training / 40 Testing)
