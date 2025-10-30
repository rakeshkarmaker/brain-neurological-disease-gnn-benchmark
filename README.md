# Brain Neurological Disease GNN Benchmark

![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This repository provides a **benchmark framework for brain neurological disease detection using Graph Neural Networks (GNNs)**. By leveraging **functional connectivity matrices** derived from fMRI data, this project explores various GNN architectures, hybrid models, and ensemble strategies to improve disease classification and **cross-site generalization**.

The goal is to create a **reproducible, extensible, and research-oriented framework** for exploring graph-based methods in neuroscience.

---

## Motivation

Neurological disease detection is challenging due to:

* Small sample sizes
* High-dimensional brain connectivity data
* Variability across imaging sites and scanners

Graph-based models can naturally capture **relationships between brain regions**, providing richer representations than traditional ML or CNN approaches.

This project benchmarks **GNNs, hybrid GNN+ML/CNN models, and ensemble techniques** to assess performance, generalizability, and robustness.

---

## Features

* Standardized **brain parcellation using the AAL atlas**.
* Functional connectivity matrix computation via **Pearson correlation**.
* Multiple GNN architectures: **GCN, GAT, GraphSAGE, etc.**
* Hybrid pipelines: **GNN + ML/CNN combinations**
* Ensemble strategies for improved generalization.
* **Cross-site evaluation** to measure robustness across different datasets.
* Preprocessing and analysis notebooks for visualization and reproducibility.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rakeshkarmaker/brain-neurological-disease-gnn-benchmark.git
cd brain-neurological-disease-gnn-benchmark

# Install dependencies
pip install -r requirements.txt
```

> Recommended: Python 3.10+ and a CUDA-enabled GPU for training.

---

## Usage

### Preprocessing

```bash
python scripts/preprocess.py --input <raw_fmri_dir> --output <processed_data_dir>
```

* Extracts mean BOLD time series from each brain region
* Computes functional connectivity matrices (edges)
* Optional normalization and thresholding for network sparsity

### Training GNN Models

```bash
python scripts/train_gnn.py --dataset <dataset_name> --model <gnn_model> --epochs 100
```

* Supports GCN, GAT, GraphSAGE, and custom GNN variants
* Configurable hyperparameters in `configs/`

### Evaluation

```bash
python scripts/evaluate.py --model <trained_model> --dataset <test_dataset>
```

* Metrics: Accuracy, AUC, F1-score
* Supports **cross-site generalization experiments**

---

## Datasets

* **AAL Atlas**: Standardized brain parcellation
* **Functional Connectivity Matrices**: Generated from fMRI time series
* **Cross-site Datasets**: Used to test model generalization across sites

> Dataset access instructions and preprocessing notes are in `docs/`

---

## Experiments & Results

* Baseline GNN models perform competitively against ML and CNN baselines
* Hybrid and ensemble models improve **cross-site generalization**
* Example summary:

| Model         | Accuracy | AUC  | F1-score |
| ------------- | -------- | ---- | -------- |
| GCN           | 0.82     | 0.85 | 0.81     |
| GAT           | 0.84     | 0.87 | 0.83     |
| ML+GNN Hybrid | 0.88     | 0.91 | 0.87     |

### Visualization & Analysis

* Functional connectivity heatmaps
* Graph embeddings visualized with t-SNE / PCA
* ROC curves and metric plots available in `notebooks/`
* Insights from experiments highlight which hybrid and ensemble configurations are most effective

---

## Project Structure

```
brain-neurological-disease-gnn-benchmark/
├── data/                  # Raw and preprocessed datasets
├── models/                # GNN/CNN/ML model definitions
├── notebooks/             # Analysis & visualization notebooks
├── scripts/               # Preprocessing and training scripts
├── docs/                  # Documentation, references, and experimental results
├── configs/               # Training and hyperparameter configs
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Future Work

* Incorporate **multi-modal data** (structural + functional MRI)
* Explore **self-supervised GNN architectures**
* Extend ensemble strategies and mixture-of-experts models
* Integration with clinical metadata for improved predictive power

---

## Contributing

Contributions are welcome!

* Fork the repository
* Create a new branch for your feature/experiment
* Submit a pull request

---

## References

* GPT AI Assisted README doc. Generated
* Automated Anatomical Labeling (AAL) Atlas
* Relevant GNN literature for neurological disease detection (see `docs/References.md`)
* Preprocessing and analysis notes in `docs/`

---

## License

MIT License – see [LICENSE](LICENSE) for details.
