## 📌 Overview

This project explores the application of **unsupervised deep learning** techniques—particularly **Autoencoders**—for detecting **stuck-at faults** in both combinational and sequential VLSI circuits. Traditional fault detection methods are often inaccurate and time-consuming, motivating the use of artificial intelligence techniques. Stuck-at faults are permanent faults that cause a signal to remain fixed at either logic high (stuck-at-1) or logic low (stuck-at-0), making the output different. Test patterns are used to identify such kinds of faults, and these are generated using Automatic Test Pattern Generation (ATPG) tools. The proposed work incorporates ATPG tools such as ATALANTA and Cadence Modus for generating test patterns and corresponding fault-free responses for various IEEE
International Symposium on Circuits and Systems (ISCAS) benchmark circuits. These test patterns are then pre-processed to be used as a dataset where an
Autoencoder algorithm classifies the faulty and non-faulty responses based on their reconstruction error.

## 📂 Folder Structure

| Folder | Description |
|--------|-------------|
| `datasets/` | Preprocessed `.csv` files for each benchmark circuit |
| `test_patterns/` | Raw test pattern files from ATALANTA and Cadence Modus |
| `models/` | Autoencoder implementations – SAE, SpAE, VAE, SSAE |
| `utils/` | Preprocessing and thresholding scripts |
| `results/` | Output plots, loss curves, and evaluation reports |

## 🤖 Autoencoder Models

- **Stacked Autoencoder (SAE)**  
- **Sparse Autoencoder (SpAE)**  
- **Variational Autoencoder (VAE)**  
- **Stacked Sparse Autoencoder (SSAE)**

Each model is trained using circuit-specific `.csv` files containing test patterns and corresponding fault-free outputs.

## 📈 Results

- Mean accuracy for **combinational circuits**: `95.14%`  
- Mean accuracy for **sequential circuits**: `96.01%`  
- Best performing model: `SAE`  
- Thresholding methods: **Percentile-based** and **Formula-based**

## 📜 Citation

If you use this code or dataset, please cite:

> Saha, C., Siddiqui, B., Bhoi, H., & Sudhanya, P. (2025). *Enhancement of Stuck-At Fault Detection in VLSI Circuits Using Deep Learning*. 3rd International Conference on Device Intelligence, Computing and Communication Technologies (DICCT). IEEE.
