# Genomic Classification via FCGR and Spiking Neural P Systems (SNPS)

A SNPS simulator for bacterial genomic classification. This project bypasses standard sliding-window k-mer extraction by using continuous Frequency Chaos Game Representation (FCGR) mapped directly into discretized matrices (4x4, 8x8, 16x16 or 32x32), followed by feature extraction and classification using SNPS.

## Project Overview

* **Input Data:** Complete genomic sequences (FASTA) from 4 bacterial genera (Aeromonas, Citrobacter, Helicobacter, Streptococcus).
* **Representation:** Direct FCGR matrix mapping, with 4 available resolutions: 4x4 (2-mers), 8x8 (3-mers), 16x16 (4-mers) and 32x32 (5-mers). The default is 8x8.
* **Architecture:** 4-Layer Spiking Neural P System with an ensemble classifier (SVM & Logistic Regression).
* **Dataset Partition:** An example with 40 total samples (30 Training / 10 Testing).

## Environment Setup

### 0. Clone the Repository

```bash
git clone https://github.com/SandroErba/SNPS_for_FCGR.git
cd SNPS_for_FCGR
```

### 1. Install Python Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install R Dependencies

Create a local R library folder:

```bash
mkdir -p ~/R/library
```

Then install the packages into it:

```bash
Rscript -e 'install.packages(c("kaos", "seqinr", "foreach", "doParallel"), lib="~/R/library", repos="https://cloud.r-project.org")'
```

## Complete Execution Pipeline

Execute the following commands in sequence to run the complete data acquisition, matrix generation, and model training workflow:

### Step 1: Download Genomic Dataset

```bash
bash download_dataset.sh
```

### Step 2: Extract FCGR Matrices

```bash
Rscript produce_fcgr.R
```

This produces 4 types of images: 4x4 (2-mers), 8x8 (3-mers), 16x16 (4-mers) and 32x32 (5-mers).

Only the first samples are extracted, for a quick test. The code contains the parameter `MAX_SAMPLES`, which can be increased to extract more data.

### Step 3: Run Training and Classification Pipeline

```bash
python main.py -i dataset_8x8
```

By default, the matrix size used is 8x8. To run with other sizes, change it in the config (`sps/config.py`).

## Configuration Settings

Dataset partition and matrix resolution parameters configured in `sps/config.py`:

* **Matrix Resolution:** 8x8 by default (`IMG_SHAPE = 8`). Available sizes: 4, 8, 16, 32.
* **Target Classes:** 4 (Aeromonas, Citrobacter, Helicobacter, Streptococcus)
* **Dataset Split:** 40 total samples (30 Training / 10 Testing)
