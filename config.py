"""
Configuration and argument parsing.
"""

import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='Input length for BERT')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=200)
    parser.add_argument('--bert_lr', type=float, default=None, help='BERT learning rate')
    parser.add_argument('--dataset', default='ohsumed', 
                        choices=['20ng', 'R8', 'R52', 'ohsumed', 'MR', 'MR_toy', 'R8_toy', 'R52_toy', 'ohsumed_toy', '20ng_toy'])
    parser.add_argument('--bert_init', type=str, default='bert-base-uncased',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'distilbert-base-uncased'])
    parser.add_argument('--checkpoint_dir', default=None, help='Checkpoint directory')
    parser.add_argument('--lam', type=float, default=None, help='Lambda parameter to weight gloss and CE')
    parser.add_argument('--gamma', type=float, default=None, help='Gamma parameter')
    parser.add_argument('--sigma', type=float, default=None, help='Sigma parameter')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'gloss', 'scl'])
    parser.add_argument('--use_latest_checkpoint', action='store_true', help='Use most recent checkpoint for model configuration')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning with Optuna')
    parser.add_argument('--sigmafn', default=None, choices=["mst", "root"])
    parser.add_argument('--optuna_trials', type=int, default=15, help='Number of Optuna trials')
    parser.add_argument('--optuna_sigma', default=None, type=float, help='Fallback value of sigma if computed sigma 0')
    parser.add_argument('--optuna_storage', type=str, default=None, help='Optuna storage URI (e.g., sqlite:///optuna_study.db)')
    parser.add_argument('--optuna_results', type=str, default='optuna_best_params.json', help='JSON file with best parameters from Optuna')
    
    return parser.parse_args()