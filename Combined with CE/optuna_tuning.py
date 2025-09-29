"""
Hyperparameter tuning with Optuna.
"""

import os
import json
import argparse

import optuna
from optuna.trial import TrialState
from torch.optim import Adam

from models import BertClassifier
from training import train_model


def objective(trial, args, train_loader, val_loader, num_labels, class_weights, device, logger):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters
    bert_lr = trial.suggest_categorical('bert_lr', [1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    
    if args.loss == 'gloss':
        if args.sigmafn is None:
            sigma = trial.suggest_float('sigma', 0.1, 10.0)
        else:
            sigma = args.sigma
        lam = trial.suggest_float('lam', 0.5, 0.9, step=0.1)
        gamma = trial.suggest_float('gamma', 0.5, 0.9, step=0.1)
        temperature = None
    elif args.loss == 'scl':
        temperature = trial.suggest_categorical('temperature', [0.1, 0.3, 0.5, 0.7])
        gamma = None
        lam = 0.9
        sigma = None
    else:
        sigma = None
        lam = None
        gamma = None
        temperature = None

    # Log trial number and hyperparameters
    logger.info(f"Trial {trial.number} started with values: bert_lr={bert_lr}, sigma={sigma}, lam={lam}, gamma={gamma}, temperature={temperature}")

    # Create a temporary args object with trial parameters
    trial_args = argparse.Namespace(**vars(args))
    trial_args.bert_lr = bert_lr
    trial_args.sigma = sigma
    trial_args.lam = lam
    trial_args.gamma = gamma
    trial_args.temperature = temperature
    
    # Initialize model
    model = BertClassifier(pretrained_model=trial_args.bert_init, nb_class=num_labels).to(device)
    optimizer = Adam(model.parameters(), lr=trial_args.bert_lr)
    
    # Run training
    model, best_val_f1 = train_model(trial_args, model, train_loader, val_loader, optimizer, 
                                    class_weights, num_labels, args.checkpoint_dir, logger, device)
    
    return best_val_f1


def run_optuna_tuning(args, train_loader, val_loader, num_labels, class_weights, device, logger):
    """Run Optuna hyperparameter tuning."""
    # Build unique db filename per dataset + loss
    db_filename = f"{args.dataset}_{args.loss}_optuna_study.db"
    storage_path = os.path.join(args.checkpoint_dir, db_filename)
    if args.optuna_storage is None:
        storage_uri = f"sqlite:///{storage_path}"
    else:
        user_input = args.optuna_storage.strip()
        if user_input.startswith(("sqlite://", "mysql://", "postgresql://")):
            storage_uri = user_input
        else:
            storage_uri = f"sqlite:///{user_input}"

    # Use a clean study name
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
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1)
        )
        logger.info(f"Created new Optuna study: {optuna_study_name} at {storage_uri}")

    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, num_labels, class_weights, device, logger), 
        n_trials=args.optuna_trials,
        gc_after_trial=True
    )

    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(args.checkpoint_dir, args.optuna_results), 'w') as f:
        json.dump(best_params, f)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (Validation F1): {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return best_params