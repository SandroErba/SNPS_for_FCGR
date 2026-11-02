import os
import argparse
from pathlib import Path
import numpy as np

# Disable TensorFlow verbose logs if present in backend
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import network
from sps.config import Config

def main():
    parser = argparse.ArgumentParser(description="SNPS Classification Pipeline for FCGR Matrices")
    parser.add_argument('-i', '--input', help='Path to the input dataset directory containing subdirectories of matrices (e.g., dataset_28x28)', required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible shuffling')
    args = parser.parse_args()

    dataset_path = Path(args.input).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"[!] The specified input directory '{dataset_path}' does not exist.")

    x = []
    y = []
    class_names = []
    
    # Sort subdirectories to guarantee deterministic class labeling
    genre_dirs = sorted([d for d in dataset_path.glob('*/') if d.is_dir()])
    
    print("==========================================================")
    print(" LOADING AND PREPROCESSING FCGR MATRICES (28x28)")
    print("==========================================================")
    
    for label_idx, genus_dir in enumerate(genre_dirs):
        genus_name = genus_dir.name
        class_names.append(genus_name)
        matrix_files = sorted(list(genus_dir.glob('*.txt')))
        
        print(f"[+] Genus '{genus_name}' (Label {label_idx}): found {len(matrix_files)} samples")
        
        for file_path in matrix_files:
            image = np.loadtxt(str(file_path), delimiter=' ')
            image = image.reshape(Config.IMG_SHAPE, Config.IMG_SHAPE)
            
            # Corrected 8-bit Min-Max scaling (0-255) for SNN input
            M, m = np.max(image), np.min(image)
            if M > m:
                image_norm = ((image - m) / (M - m)) * 255.0
            else:
                image_norm = np.zeros_like(image)
                
            x.append(image_norm.astype(int).tolist())
            y.append(label_idx)

    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)

    # Reproducible random permutation of dataset
    np.random.seed(args.seed)
    permutation = np.random.permutation(len(x))
    x = x[permutation]
    y = y[permutation]

    print(f"\n[+] Dataset successfully loaded: {x.shape[0]} total samples at resolution {Config.IMG_SHAPE}x{Config.IMG_SHAPE}")
    print(f"[+] Identified classes: {class_names}")
    
    Config.compute_k_range()
    
    print("\n==========================================================")
    print(" STARTING SNPS TRAINING AND EVALUATION")
    print("==========================================================")
    network.launch_dataset(x, y)

if __name__ == '__main__':
    main()