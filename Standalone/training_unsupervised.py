"""
Training functions for unsupervised learning.
"""

import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ignite.engine import Events, Engine
import optuna

from losses_unsupervised import (
    predict_lpa,
    supervised_contrastive_loss,
    BatchAllTripletLoss,
    BatchCosineSimilarityLoss
)
from metrics_unsupervised import UnsupervisedMetricsTracker


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model for a given dataloader."""
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask)
            emb = outputs[0][:, 0]  # CLS token embedding
            embeddings.append(emb)
            labels_list.append(labels)
    
    return torch.cat(embeddings), torch.cat(labels_list)


def train_linear_classifier(model, train_emb, train_labels, val_emb, val_labels, 
    num_labels, class_weights, device, lr=0.0001, patience=10, max_epochs=50, ckpt_dir=None, logger=None):
    """
    Train a linear classifier on top of frozen embeddings.
    
    Args:
        model: Base model (not used, kept for compatibility)
        train_emb: Training embeddings
        train_labels: Training labels
        val_emb: Validation embeddings
        val_labels: Validation labels
        num_labels: Number of classes
        class_weights: Class weights for imbalanced datasets
        device: PyTorch device
        lr: Learning rate
        patience: Early stopping patience
        max_epochs: Maximum training epochs
        ckpt_dir: Directory to save classifier checkpoint
        logger: Logger instance
    
    Returns:
        Best validation F1 score
    """
    classifier = nn.Linear(train_emb.size(1), num_labels).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    best_val_f1 = 0
    best_epoch = 0
    counter = 0
    best_class_state = None    
    start_time = time.time()
    epoch_time = []
    
    # Convert class weights to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    for epoch in range(max_epochs):
        e1 = time.time()
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_emb)
        loss = F.cross_entropy(logits, train_labels, weight=class_weights_tensor)
        loss.backward()
        optimizer.step()
        
        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_emb)
            val_preds = torch.argmax(val_logits, dim=1)
            val_f1 = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='macro') 
        
        # Early stopping logic
        if val_f1 > best_val_f1:  # Track best macro F1
            best_val_f1 = val_f1
            best_epoch = epoch
            counter = 0
            best_class_state = classifier.state_dict().copy()
        else:
            counter += 1
        
        if counter >= patience:
            break
        e2 = time.time()
        epoch_time.append(e2-e1)

    if logger:
        logger.info(f"Early Stopping at LC: {best_epoch}")
        logger.info(f"Total training time LC: {time.time()-start_time}")
        logger.info(f"Avg time per epoch LC: {sum(epoch_time)/len(epoch_time) if len(epoch_time) > 0 else 0:.2f}")
        logger.info(f"Macro best_val_f1: {best_val_f1}")

    if best_class_state is not None:
        classifier.load_state_dict(best_class_state)
        if ckpt_dir:
            torch.save({
                'lc': classifier.state_dict(),
            }, os.path.join(ckpt_dir, 'classifier.pth'))
    
    return best_val_f1


def run_training(args, train_loader, val_loader, num_labels, class_weights, bert_lr, gamma, sigma, temperature, device, logger, trial=None):
    """
    Main training loop for unsupervised learning.
    
    Args:
        args: Command line arguments
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_labels: Number of classes
        class_weights: Class weights for imbalanced datasets
        bert_lr: Learning rate for BERT
        gamma: Gamma parameter for G-Loss
        sigma: Sigma parameter for G-Loss
        temperature: Temperature for contrastive loss
        device: PyTorch device
        logger: Logger instance
        trial: Optuna trial object (if tuning)
    
    Returns:
        Tuple of (model, best_silhouette)
    """
    from transformers import AutoModel
    
    total_start_time = time.time()
    early_stop_epoch = None
    epoch_times = []
    train_losses = []
    
    model = AutoModel.from_pretrained(args.bert_init).to(device)
    hidden_size = model.config.hidden_size
    
    if args.loss == 'cross_entropy':
        model.classifier = nn.Linear(hidden_size, num_labels).to(device)
        
    # Loss function selection
    def get_loss(input_ids, attention_mask, labels, model):
        if args.loss == 'cross_entropy':
            logits = model.classifier(model(input_ids, attention_mask).last_hidden_state[:, 0, :])
            return F.cross_entropy(
                logits, 
                labels,
                weight=torch.tensor(class_weights, dtype=torch.float64).to(device)
            )
        elif args.loss == 'g-loss':
            return predict_lpa(
                model, input_ids, attention_mask, labels,
                sigma or 0.1, num_labels, 
                gamma or 0.7, device, class_weights
            )
        elif args.loss == 'scl':
            return supervised_contrastive_loss(
                model, input_ids, attention_mask, labels, 
                temperature or 0.1
            )
        elif args.loss == 'triplet':
            return BatchAllTripletLoss(model)(
                input_ids, attention_mask, labels
            )
        elif args.loss == 'cos-sim':
            return BatchCosineSimilarityLoss(model)(
                input_ids, attention_mask, labels
            )
    
    # Training step function
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        loss = get_loss(input_ids, attention_mask, labels, model)
        loss.backward()

        train_losses.append(loss.item())
        
        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return float('inf')  # This will help Optuna prune this trial
        
        # Use a less aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        return loss.item()

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    
    # Setup Ignite trainer
    trainer = Engine(train_step)
    ckpt_dir = args.checkpoint_dir if not trial else None
    metrics_tracker = UnsupervisedMetricsTracker(model, ckpt_dir, num_labels, args=args, device=device)
    best_silhouette = -1
    best_model_state = None
    
    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch(engine):
        engine.state.epoch_start = time.time()
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch(engine):
        epoch = engine.state.epoch
        epoch_time = time.time() - engine.state.epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        epoch_times.append(epoch_time)  # Track epoch time
        
        avg_loss = np.mean(train_losses) if train_losses else 0.0
        is_best, early_stop, sil_score = metrics_tracker.evaluate_epoch(epoch, val_loader, avg_loss, logger)
        metrics_tracker.save_epoch_stats_to_csv()
        train_losses.clear()  # Reset training losses for next epoch

        # Track best silhouette score
        nonlocal best_silhouette, best_model_state, early_stop_epoch
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_model_state = model.state_dict().copy()
        
        # For Optuna trials, report intermediate value
        if trial:
            trial.report(sil_score, epoch)
            if trial.should_prune():
                engine.terminate()
                raise optuna.exceptions.TrialPruned()
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping
        if early_stop:
            early_stop_epoch = epoch
            engine.terminate()
    
    # Run training
    try:
        trainer.run(train_loader, max_epochs=args.nb_epochs)
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial pruned after {len(epoch_times)} epochs")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    total_time = time.time() - total_start_time

    logger.info(f"Total time: {total_time}")
    logger.info(f"Avg epoch time: {sum(epoch_times)/len(epoch_times) if len(epoch_times) > 0 else 0:.2f}")
    logger.info(f"Early stop epoch: {early_stop_epoch}")

    return model, best_silhouette