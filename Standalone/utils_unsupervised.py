"""
Utility functions for data loading, preprocessing, and helper functions.
"""

import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


def setup_logger(ckpt_dir):
    """Setup logger for training."""
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    
    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    
    # File handler
    fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    print(f"Logging to {os.path.join(ckpt_dir, 'training.log')} ")
    return logger


def find_latest_checkpoint(base_dir='./macro_checkpoint', loss_function=None, bert_init=None, data=None):
    """Find the most recent checkpoint directory."""
    pattern_parts = [loss_function, bert_init, data]
    pattern = "_".join([p for p in pattern_parts if p]) + "_*"
    search_path = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_path)
    
    if not matching_dirs:
        return None
        
    timestamp_format = "%Y-%m-%d-%H-%M-%S"
    def extract_timestamp(dir_path):
        dir_name = os.path.basename(dir_path)
        timestamp_str = dir_name.split('_')[-1]
        try:
            return datetime.strptime(timestamp_str, timestamp_format)
        except ValueError:
            return datetime(1900, 1, 1)
    
    return sorted(matching_dirs, key=extract_timestamp, reverse=True)[0]


def load_data(data_dir):
    """Load train, validation, and test datasets."""
    train = pd.read_csv(os.path.join(data_dir, 'train.csv')).sample(frac=1).reset_index(drop=True)
    val = pd.read_csv(os.path.join(data_dir, 'val.csv')).sample(frac=1).reset_index(drop=True)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv')).sample(frac=1).reset_index(drop=True)
    return train, val, test


def compute_class_weights(labels):
    """Compute balanced class weights."""
    return compute_class_weight('balanced', classes=np.unique(labels), y=labels)


def encode_data(df, tokenizer, max_length):
    """Encode text data using tokenizer."""
    encoded = tokenizer(
        list(df["text"]), 
        max_length=max_length, 
        truncation=True, 
        padding="max_length", 
        return_tensors='pt'
    )
    return encoded.input_ids, encoded.attention_mask


def prepare_dataloaders(args, train, val, test):
    """Prepare PyTorch dataloaders for training."""
    tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
    
    # Encode datasets
    train_input_ids, train_attention_mask = encode_data(train, tokenizer, args.max_length)
    val_input_ids, val_attention_mask = encode_data(val, tokenizer, args.max_length)
    test_input_ids, test_attention_mask = encode_data(test, tokenizer, args.max_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = torch.tensor(label_encoder.fit_transform(train['label']), dtype=torch.long)
    val_labels = torch.tensor(label_encoder.transform(val['label']), dtype=torch.long)
    test_labels = torch.tensor(label_encoder.transform(test['label']), dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(label_encoder.classes_)


def compute_sigma(X: torch.Tensor, labels: torch.Tensor, sigmafn: str, logger) -> float:
    """
    Compute sigma using the Direct Inter-Class Minimum (DICM) method.

    This method finds the minimum distance between any two points that have
    different labels. If the minimum distance is zero, it selects the next smallest 
    non-zero distance. If no non-zero distances exist, it falls back to the median 
    distance between all points.

    Args:
        X (torch.Tensor): Embedding tensor of shape (N, D).
        labels (torch.Tensor): Corresponding labels of shape (N,).
                               Unlabeled points should be marked with -1.
        sigmafn (str): Sigma function type ('root' or 'mst').
        logger: Logger instance for info messages.

    Returns:
        Tuple[float, float]: The calculated DICM sigma value (d0 / 3) and Henil sigma value (sqrt(d_henil / 3)).
    """
    device = X.device
    n = X.shape[0]

    # Precompute the full distance matrix once
    dist_matrix = torch.cdist(X, X, p=2.0)
    d_henil = torch.median(dist_matrix)  # For Henil sigma (entire matrix including diagonal)
    sigma_henil = torch.sqrt(d_henil / 3).item()

    if sigmafn == 'root':
        return 0, sigma_henil
    else:
        # --- Step 1: Create masks ---
        final_mask = (labels.view(-1, 1) != labels.view(1, -1))

        # --- Step 2: Handle no inter-class pairs ---
        if not torch.any(final_mask):
            logger.info("DICM Fallback: No inter-class pairs found. Using median heuristic.")
            upper_triangle_indices = torch.triu_indices(n, n, offset=1, device=device)
            d0_fallback = torch.median(dist_matrix[upper_triangle_indices[0], upper_triangle_indices[1]])
            return (d0_fallback / 3).item(), sigma_henil

        # --- Step 3: Compute inter-class distances ---
        inter_class_distances = torch.where(final_mask, dist_matrix, torch.tensor(float('inf'), device=device))
        d0 = torch.min(inter_class_distances)

        # --- Step 4: Handle zero minimum distance ---
        if d0 == 0:
            # Extract finite non-zero distances
            non_zero_distances = inter_class_distances[inter_class_distances > 0]
            if non_zero_distances.numel() > 0:
                d0 = torch.min(non_zero_distances)
            else:
                logger.info("DICM Fallback: All inter-class distances are zero. Using median heuristic.")
                upper_triangle_indices = torch.triu_indices(n, n, offset=1, device=device)
                d0_fallback = torch.median(dist_matrix[upper_triangle_indices[0], upper_triangle_indices[1]])
                return (d0_fallback / 3).item(), sigma_henil

        sigma_dicm = d0 / 3
        return sigma_dicm.item(), sigma_henil


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model for a given dataloader."""
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            _, outputs = model(input_ids, attention_mask)
            embeddings.append(outputs)
            labels_list.append(labels)
    
    return torch.cat(embeddings), torch.cat(labels_list)