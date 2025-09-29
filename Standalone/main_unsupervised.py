"""
Main training script for unsupervised learning.
"""

import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, classification_report
from transformers import AutoModel

# Reuse common utilities from supervised version
from utils import (
    setup_logger,
    find_latest_checkpoint,
    load_data,
    compute_class_weights,
    prepare_dataloaders
)

# Unsupervised-specific modules
from config_unsupervised import parse_arguments
from utils_unsupervised import compute_sigma
from training_unsupervised import run_training, extract_embeddings, train_linear_classifier
from optuna_tuning_unsupervised import run_optuna_tuning
from metrics_unsupervised import UnsupervisedMetricsTracker, macro_sil_score


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_log = list()


def main():
    args = parse_arguments()
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Setup checkpoint directory
    if args.checkpoint_dir is None and args.use_latest_checkpoint:
        ckpt_dir = find_latest_checkpoint(
            loss_function=args.loss, 
            bert_init=args.bert_init, 
            data=args.dataset
        ) 
        if ckpt_dir is None:
            suffix = "_root" if args.sigmafn == "root" else ""
            ckpt_dir = f"./macro_checkpoint/{args.loss}_{args.bert_init}_{args.dataset}{suffix}_{now}"
    elif args.checkpoint_dir is None:
        suffix = "_root" if args.sigmafn == "root" else ""
        ckpt_dir = f"./macro_checkpoint/{args.loss}_{args.bert_init}_{args.dataset}{suffix}_{now}"
    else:
        ckpt_dir = args.checkpoint_dir

    global logger
    logger = setup_logger(ckpt_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    args.checkpoint_dir = ckpt_dir    

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    data_dir = f"data/{args.dataset}/"
    train, val, test = load_data(data_dir)
    train_loader, val_loader, test_loader, num_labels = prepare_dataloaders(args, train, val, test)
    class_weights = compute_class_weights(train['label'])

    model = AutoModel.from_pretrained(args.bert_init).to(device)
    hidden_size = model.config.hidden_size

    if args.sigmafn is not None:
        logger.info(f"Using sigma function: {args.sigmafn}")
        train_emb, train_labels = extract_embeddings(model, train_loader, device)
        train_emb = train_emb / train_emb.norm(dim=1, keepdim=True)
        sigma_heu, sigma_henil = compute_sigma(train_emb, train_labels, logger)
        args.sigma = sigma_heu if args.sigmafn == 'mst' else sigma_henil
        logger.info(f"Computed sigma: {args.sigma}, using {args.sigmafn}")
        if sigma_heu < 0.01:
            args.sigma = args.optuna_sigma
            logger.info(f"using fallback sigma from optuna {args.sigma} ")

    # Run hyperparameter tuning if requested
    if args.tune:
        logger.info("Starting Optuna hyperparameter tuning...")
        best_params = run_optuna_tuning(args, train_loader, val_loader, num_labels, class_weights, device, logger)
        
        # Update args with best parameters
        args.bert_lr = best_params.get('bert_lr', args.bert_lr)
        if args.loss == 'g-loss':
            args.gamma = best_params.get('gamma', args.gamma)
            args.sigma = best_params.get('sigma', args.sigma)
        if args.loss == 'scl':
            args.temperature = best_params.get('temperature', args.temperature)
        
        logger.info(f"Using best parameters: bert_lr={args.bert_lr}, gamma={args.gamma}, sigma={args.sigma}, temperature={args.temperature}")
    
    # Initialize model
    model = AutoModel.from_pretrained(args.bert_init).to(device)
    hidden_size = model.config.hidden_size
    
    # Run main training with best parameters
    logger.info("Starting main training...")
    global batch_log
    batch_log.clear()
    
    model, best_silhouette = run_training(
        args, train_loader, val_loader, num_labels, class_weights, 
        args.bert_lr or 3e-5, args.gamma or 0.7, sigma=args.sigma or 0.1, 
        temperature=args.temperature, device=device, logger=logger
    )
    logger.info(f"Best silhouette score: {best_silhouette:.4f}")
    
    # Save best model
    torch.save({
        'model': model.state_dict(),
        'silhouette': best_silhouette
    }, os.path.join(ckpt_dir, 'best_model.pth'))
    torch.save(batch_log, os.path.join(ckpt_dir, 'batch_graph.pt')) 
    
    # Final evaluation
    evaluator = UnsupervisedMetricsTracker(model, ckpt_dir, num_labels, args=args, device=device)
    evaluator.evaluate_epoch(0, test_loader, loss=0.0, logger=logger)  # Create t-SNE plot
    
    # Train and evaluate linear classifier
    logger.info("Training final linear classifier...")
    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    val_emb, val_labels = extract_embeddings(model, val_loader, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    
    # Save embeddings
    torch.save(train_emb, os.path.join(ckpt_dir, 'train_emb.pt'))
    torch.save(train_labels, os.path.join(ckpt_dir, 'train_labels.pt'))
    torch.save(val_emb, os.path.join(ckpt_dir, 'val_emb.pt'))
    torch.save(val_labels, os.path.join(ckpt_dir, 'val_labels.pt'))
    torch.save(test_emb, os.path.join(ckpt_dir, 'test_emb.pt'))
    torch.save(test_labels, os.path.join(ckpt_dir, 'test_labels.pt'))
    
    linear_lr = 1e-04
    
    best_val_f1 = train_linear_classifier(
        model, train_emb, train_labels, val_emb, val_labels,
        num_labels, class_weights, device, linear_lr, ckpt_dir=ckpt_dir, logger=logger
    )
    
    # Test evaluation
    classifier = nn.Linear(hidden_size, num_labels).to(device)
    classifier.load_state_dict(torch.load(os.path.join(ckpt_dir, 'classifier.pth'))['lc'])
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_emb)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = accuracy_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
        test_f1 = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy(), average='macro')
    
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    macro_silhouette = macro_sil_score(test_emb.cpu().numpy(), test_labels.cpu().numpy()) if len(np.unique(test_labels.cpu().numpy())) > 1 else 0
    micro_silhouette = silhouette_score(test_emb.cpu().numpy(), test_labels.cpu().numpy(), metric='euclidean') if len(np.unique(test_labels.cpu().numpy())) > 1 else 0
    logger.info(f"Test macro SilScore: {macro_silhouette:.4f}")
    logger.info(f"Test micro SilScore: {micro_silhouette:.4f}")
    logger.info(str(classification_report(test_labels.cpu().numpy(), test_preds.cpu().numpy())))
    
    log_output = f"""\nFinal Test Results:\nTest Accuracy: {test_acc:.4f}\nTest Macro F1: {test_f1:.4f}\n\nClassification Report:\n{classification_report(test_labels.cpu().numpy(), test_preds.cpu().numpy())}"""
    logger.info(log_output)
    log_file = os.path.join(ckpt_dir, "test.log")
    open(log_file, "w").write(log_output)
       

if __name__ == "__main__":
    main()