"""
Hyperparameter tuning with Optuna for unsupervised training.
"""

import os
import json
import optuna
from training_unsupervised import run_training


def objective(trial, args, train_loader, val_loader, num_labels, class_weights, device, logger):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters
    bert_lr = trial.suggest_categorical('bert_lr', [1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    
    if args.loss == 'g-loss':
        if args.sigmafn is None:
            sigma = trial.suggest_float('sigma', 0.1, 10.0)
            gamma = trial.suggest_float('gamma', 0.5, 0.9, step=0.1)
        else:
            sigma = args.sigma
            gamma = trial.suggest_float('gamma', 0.5, 0.9, step=0.1)
        temperature = None
    elif args.loss == 'scl':
        temperature = trial.suggest_float('temperature', 0.1, 1.0)
        gamma = None  
        sigma = None
    else:
        gamma = None  
        sigma = None
        temperature = None

    # Log trial number and hyperparameters
    logger.info(f"Trial {trial.number} started with values: bert_lr={bert_lr}, sigma={sigma}, gamma={gamma}, temperature={temperature}")

    # Run BERT training
    model, best_silhouette = run_training(
        args, train_loader, val_loader, num_labels, class_weights,
        bert_lr, gamma, sigma, temperature, device, logger, trial
    )
    
    return best_silhouette


def run_optuna_tuning(args, train_loader, val_loader, num_labels, class_weights, device, logger):
    """Run Optuna hyperparameter tuning for unsupervised training."""
    seed = 42
    # Build unique db filename per dataset + loss
    db_filename = f"{args.dataset}_{args.loss}_optuna_study.db"
    storage_path = os.path.join(args.checkpoint_dir, db_filename)
    storage_uri = f"sqlite:///{storage_path}" if args.optuna_storage is None else args.optuna_storage

    # Use a clean study name (no .db inside the name)
    optuna_study_name = f"{args.dataset}_{args.loss}_optuna_study"

    try:
        study = optuna.load_study(
            study_name=optuna_study_name,
            storage=storage_uri
        )
        logger.info(f"Loaded existing Optuna study: {optuna_study_name} at {storage_uri}")
    except KeyError:
        study = optuna.create_study(
            study_name=optuna_study_name,
            storage=storage_uri,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed), 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1)
        )
        logger.info(f"Created new Optuna study: {optuna_study_name} at {storage_uri}")

    if args.loss in ['triplet', 'cos-sim']:
        args.optuna_trials = 5
        
    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, num_labels, class_weights, device, logger), 
        n_trials=args.optuna_trials,
        gc_after_trial=True,
        show_progress_bar=True
    )

    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(args.checkpoint_dir, args.optuna_results), 'w') as f:
        json.dump(best_params, f)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (Validation Sil score): {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return best_params