## Regulatory DNA Prediction from Peak and Context Sequences Using BiLSTMs and Transformers

A deep learning project investigating whether genomic context alone contains sufficient information to predict regulatory DNA activity.

This project compares recurrent and transformer-based neural networks trained on genomic sequences surrounding regulatory regions. The goal is to determine how much predictive signal exists outside of the regulatory element itself.

## Project Overview

Regulatory DNA controls gene expression through promoters, enhancers, and other functional elements. While many computational models focus on the regulatory sequence itself, this project investigates whether the surrounding genomic context contains enough information to predict regulatory activity.

Models were trained using ENCODE DNase-seq peak data and the human reference genome (GRCh38).

## Models

- BiLSTM
- Transformer Encoder

## Dataset

Positive examples were generated from DNase-seq peaks.

Negative examples were sampled from non-overlapping genomic regions.

Reference genome:

- Human Genome GRCh38

## Repository Structure

```
.
├── data/
│   └── README.md
├── notebooks/
├── src/
├── results/
├── figures/
└── README.md
```

## Results

The project compares BiLSTM and Transformer architectures for binary classification of regulatory versus non-regulatory genomic regions.

Performance is evaluated using:

- Accuracy
- ROC-AUC
- Validation loss

## Data Availability

Large genomic datasets are **not included** in this repository due to GitHub's file size limitations.

See `data/README.md` for instructions on obtaining the required files.