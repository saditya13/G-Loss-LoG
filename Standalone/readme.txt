# Unsupervised Text Classification with Contrastive Learning

This repository contains a modular implementation of unsupervised text classification using BERT-based models with various loss functions including Supervised Contrastive Learning (SCL), Graph-based Loss (G-Loss), Triplet Loss, and Cosine Similarity Loss.

## Project Structure

```
.
├── main_unsupervised.py              # Main training script for unsupervised learning
├── config_unsupervised.py            # Configuration and argument parsing
├── losses_unsupervised.py            # Loss functions (G-Loss, SCL, Triplet, Cosine)
├── training_unsupervised.py          # Training and evaluation functions
├── metrics_unsupervised.py           # Unsupervised metrics (Silhouette, t-SNE)
├── utils_unsupervised.py             # Unsupervised-specific utilities (sigma computation)
├── optuna_tuning_unsupervised.py     # Hyperparameter tuning with Optuna
├── requirements_unsupervised.txt     # Python dependencies
└── README_unsupervised.md            # This file

# Shared modules from supervised version
├── utils.py                          # Common utilities (data loading, preprocessing)
├── models.py                         # Model architectures (if needed)
```

## Key Differences from Supervised Version

- **Unsupervised Training**: Uses labels only for validation metrics (silhouette score)
- **Linear Classifier**: Trains a separate linear classifier on frozen embeddings after representation learning
- **Metrics**: Uses silhouette score instead of accuracy/F1 during training
- **Visualization**: Generates t-SNE plots during training to visualize clustering

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements_unsupervised.txt
```

## Usage

### Basic Training with SCL

```bash
python main_unsupervised.py --dataset ohsumed --loss scl --temperature 0.5
```

### Training with G-Loss

```bash
python main_unsupervised.py --dataset ohsumed --loss g-loss --gamma 0.7 --sigma 1.0
```

### Training with Triplet Loss

```bash
python main_unsupervised.py --dataset ohsumed --loss triplet
```

### Training with Cosine Similarity Loss

```bash
python main_unsupervised.py --dataset ohsumed --loss cos-sim
```

### Hyperparameter Tuning with Optuna

```bash
python main_unsupervised.py --dataset ohsumed --loss scl --tune --optuna_trials 20
```

### Using Computed Sigma for G-Loss

```bash
# Using MST method
python main_unsupervised.py --dataset ohsumed --loss g-loss --sigmafn mst

# Using root method
python main_unsupervised.py --dataset ohsumed --loss g-loss --sigmafn root
```

## Arguments

### General Arguments
- `--dataset`: Dataset name (choices: 20ng, R8, R52, ohsumed, MR)
- `--bert_init`: BERT model initialization (default: bert-base-uncased)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size (default: 128)
- `--nb_epochs`: Number of training epochs (default: 200)
- `--bert_lr`: Learning rate for BERT (default: None, will be tuned)

### Loss Function Arguments
- `--loss`: Loss function type (choices: cross_entropy, g-loss, scl, triplet, cos-sim)
- `--gamma`: Gamma parameter for G-Loss (default: None)
- `--sigma`: Sigma parameter for G-Loss (default: None)
- `--temperature`: Temperature for SCL loss (default: None)
- `--sigmafn`: Sigma computation method (choices: mst, root)

### Optuna Arguments
- `--tune`: Enable hyperparameter tuning
- `--optuna_trials`: Number of Optuna trials (default: 15, reduced to 5 for triplet/cos-sim)
- `--optuna_sigma`: Fallback sigma value (default: None)
- `--optuna_storage`: Optuna storage URI
- `--optuna_results`: JSON file for best parameters

### Other Arguments
- `--checkpoint_dir`: Checkpoint directory (default: None)
- `--use_latest_checkpoint`: Use most recent checkpoint

## Data Format

The code expects data in the following format:
- CSV files with columns: `text`, `label`
- Directory structure:
  ```
  data/
  └── <dataset_name>/
      ├── train.csv
      ├── val.csv
      └── test.csv
  ```

**Note**: Labels are used only for validation metrics (silhouette score) and final linear classifier evaluation.

## Output

Training outputs are saved in the checkpoint directory:
- `training.log`: Training logs
- `best_model.pth`: Best model based on silhouette score
- `classifier.pth`: Linear classifier trained on frozen embeddings
- `tsne_epoch_*.png`: t-SNE visualizations for each epoch
- `epoch_stats.csv`: Per-epoch statistics (loss, silhouette)
- `train_emb.pt`, `val_emb.pt`, `test_emb.pt`: Saved embeddings
- `train_labels.pt`, `val_labels.pt`, `test_labels.pt`: Saved labels
- `test.log`: Final test results

## Module Descriptions

### losses_unsupervised.py
Implements:
- G-Loss (Graph-based Loss) with Label Propagation
- Supervised Contrastive Loss (SupConLoss)
- Batch All Triplet Loss (from sentence-transformers)
- Batch Cosine Similarity Loss
- Helper functions for adjacency matrix normalization

### training_unsupervised.py
Contains:
- Main training loop using PyTorch Ignite
- Linear classifier training on frozen embeddings
- Embedding extraction utilities

### metrics_unsupervised.py
Implements:
- Macro silhouette score computation
- UnsupervisedMetricsTracker class for tracking metrics
- t-SNE visualization generation
- Early stopping based on silhouette score

### utils_unsupervised.py
Utility functions for:
- Sigma computation (DICM method)
- (Reuses common utilities from utils.py)

### optuna_tuning_unsupervised.py
Hyperparameter optimization:
- Objective function for unsupervised training
- Study creation and management
- Best parameter selection based on silhouette score

### config_unsupervised.py
Command-line argument parsing and configuration for unsupervised training.

## Training Pipeline

1. **Representation Learning**: Train BERT model with contrastive/graph-based losses
2. **Silhouette Evaluation**: Monitor clustering quality using silhouette score
3. **Linear Classifier**: Train a linear classifier on frozen embeddings
4. **Final Evaluation**: Evaluate on test set using the linear classifier

## Key Metrics

- **Macro Silhouette Score**: Average silhouette score across all classes
- **Micro Silhouette Score**: Overall silhouette score
- **Test Accuracy**: Classification accuracy on test set (after linear classifier)
- **Test Macro F1**: F1 score on test set (after linear classifier)

## Citation

If you use this code, please cite the relevant papers for:
- BERT: [Devlin et al., 2019]
- Supervised Contrastive Learning: [Khosla et al., 2020]
- Sentence Transformers: [Reimers & Gurevych, 2019]
- Your paper (if applicable)

## License

[Add your license information here]