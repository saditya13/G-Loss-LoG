# Graph-inspired fine-tuning of Language Models

This repository contains a modular implementation of supervised text classification using BERT-based models with G-Loss functions and other loss functions, including Cross-Entropy (CE), and Supervised Contrastive Learning (SCL).

## Project Structure

```
.
├── main.py                 # Main training script
├── config.py              # Configuration and argument parsing
├── models.py              # Model architectures
├── losses.py              # Loss functions (G-Loss, SCL)
├── training.py            # Training and evaluation functions
├── utils.py               # Utility functions for data loading and preprocessing
├── optuna_tuning.py       # Hyperparameter tuning with Optuna
├── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train a model with Cross-Entropy loss:
```bash
python main.py --dataset ohsumed --loss ce --bert_lr 2e-5
```

### Training with G-Loss

Train with Graph-based Loss:
```bash
python main.py --dataset ohsumed --loss gloss --lam 0.5 --gamma 0.7 --sigma 1.0
```

### Training with Supervised Contrastive Loss

```bash
python main.py --dataset ohsumed --loss scl --temperature 0.1
```

### Hyperparameter Tuning with Optuna

```bash
python main.py --dataset ohsumed --loss gloss --tune --optuna_trials 20
```

### Using Computed Sigma

For G-Loss, you can automatically compute sigma using different methods:

```bash
# Using root method
python main.py --dataset ohsumed --loss gloss --sigmafn root
```

## Arguments

### General Arguments
- `--dataset`: Dataset name (choices: 20ng, R8, R52, ohsumed, MR)
- `--bert_init`: BERT model initialization (default: bert-base-uncased)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size (default: 128)
- `--nb_epochs`: Number of training epochs (default: 200)
- `--bert_lr`: Learning rate for BERT (default: None)

### Loss Function Arguments
- `--loss`: Loss function type (choices: ce, gloss, scl)
- `--lam`: Lambda parameter for G-Loss (default: None)
- `--gamma`: Gamma parameter for G-Loss (default: None)
- `--sigma`: Sigma parameter for G-Loss (default: None)
- `--temperature`: Temperature parameter for SCL (default: 0.1)
- `--sigmafn`: Sigma computation method (choices: mst, root)

### Optuna Arguments
- `--tune`: Enable hyperparameter tuning
- `--optuna_trials`: Number of Optuna trials (default: 15)
- `--optuna_sigma`: Fallback sigma value (default: None)
- `--optuna_storage`: Optuna storage URI (default: None)
- `--optuna_results`: JSON file for best parameters (default: optuna_best_params.json)

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

## Output

Training outputs are saved in the checkpoint directory:
- `training.log`: Training logs
- `checkpoint.pth`: Model checkpoints
- `best_model.pth`: Best model based on validation F1
- `train_loss.png`: Training loss plot
- `val_f1.png`: Validation F1 plot
- `epoch_stats.csv`: Per-epoch statistics
- `loss_stats.csv`: Loss component statistics

## Module Descriptions

### models.py
Contains the `BertClassifier` class for text classification.

### losses.py
Implements:
- G-Loss (Graph-based Loss) with Label Propagation
- Supervised Contrastive Loss
- Helper functions for adjacency matrix normalization

### training.py
Contains training loop and evaluation functions.

### utils.py
Utility functions for:
- Data loading and preprocessing
- Tokenization and encoding
- Sigma computation
- Embedding extraction

### optuna_tuning.py
Hyperparameter optimization using Optuna framework.

### config.py
Command-line argument parsing and configuration.

## Citation

If you use this code, please cite the relevant papers for:
- Your paper (if applicable)

## License

[Add your license information here]
