"""
Main training script.
"""

import os
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam

from config import parse_arguments
from utils import (
    setup_logger, 
    find_latest_checkpoint, 
    load_data, 
    compute_class_weights,
    prepare_dataloaders,
    compute_sigma,
    extract_embeddings
)
from models import BertClassifier
from training import train_model, test_evaluate
from optuna_tuning import run_optuna_tuning


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_arguments()
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Setup checkpoint directory
    suffix = ""
    if args.checkpoint_dir is None and args.use_latest_checkpoint:
        ckpt_dir = find_latest_checkpoint(
            loss_function=args.loss,
            bert_init=args.bert_init,
            data=args.dataset
        )
        if ckpt_dir is None:
            if args.loss == 'gloss' and args.sigmafn == 'root':
                suffix = "_root"
            elif args.loss == 'gloss' and args.sigmafn == 'mst':
                suffix = "_mst"
            ckpt_dir = f"./supervised_checkpoint/{args.loss}_{args.bert_init}_{args.dataset}{suffix}_{now}"
    elif args.checkpoint_dir is None:
        if args.loss == 'gloss' and args.sigmafn == 'root':
            suffix = "_root"
        elif args.loss == 'gloss' and args.sigmafn == 'mst':
            suffix = "_mst"
        ckpt_dir = f"./supervised_checkpoint/{args.loss}_{args.bert_init}_{args.dataset}{suffix}_{now}"
    else:
        ckpt_dir = args.checkpoint_dir
        
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
    logger.info(f"Train samples: {len(train)} | Val samples: {len(val)} | Test samples: {len(test)}")

    train_loader, val_loader, test_loader, num_labels = prepare_dataloaders(args, train, val, test)
    class_weights = compute_class_weights(train['label']) 

    # Instantiate model according to class number
    model = BertClassifier(pretrained_model=args.bert_init, nb_class=num_labels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized: {model.__class__.__name__} | Total parameters: {total_params}")
    
    if args.sigmafn is not None:
        logger.info(f"Using sigma function: {args.sigmafn}")
        train_emb, train_labels = extract_embeddings(model, train_loader, device)
        train_emb = train_emb / train_emb.norm(dim=1, keepdim=True)
        sigma_heu, sigma_henil = compute_sigma(train_emb, train_labels, args.sigmafn, logger)
        args.sigma = sigma_heu if args.sigmafn == 'mst' else sigma_henil
        logger.info(f"Computed sigma: {args.sigma}, using {args.sigmafn}")
        if args.sigma < 0.01:
            args.sigma = args.optuna_sigma
            logger.info(f"using fallback sigma from optuna {args.sigma} ")

    if args.tune:
        logger.info("Starting Optuna hyperparameter tuning...")
        best_params = run_optuna_tuning(args, train_loader, val_loader, num_labels, class_weights, device, logger)
        
        # Update args with best parameters
        args.bert_lr = best_params.get('bert_lr', args.bert_lr)
        if args.loss == 'gloss':
            args.lam = best_params.get('lam', args.lam)
            args.gamma = best_params.get('gamma', args.gamma)
            args.sigma = best_params.get('sigma', args.sigma)
        
        logger.info(f"Using best parameters: bert_lr={args.bert_lr}, lam={args.lam}, gamma={args.gamma}, sigma={args.sigma}")
    
    # Run main training with best parameters
    logger.info("Starting main training...")
    optimizer = Adam(model.parameters(), lr=args.bert_lr)

    model, best_val_f1 = train_model(args, model, train_loader, val_loader, optimizer, class_weights, num_labels, ckpt_dir, logger, device)
    logger.info(f"Best validation F1 score: {best_val_f1:.4f}")
    
    # Save best model
    torch.save({
        'model': model.state_dict(),
        'best_val_f1': best_val_f1
    }, os.path.join(ckpt_dir, 'best_model.pth'))

    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth'))['model'])
    test_evaluate(model, test_loader, logger, device)


if __name__ == "__main__":
    main()