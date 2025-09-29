"""
Metrics and evaluation for unsupervised learning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE


def macro_sil_score(embeddings, pred_labels):
    """
    Compute macro-averaged silhouette score.
    
    Args:
        embeddings: Embedding vectors
        pred_labels: Predicted cluster labels
    
    Returns:
        Macro-averaged silhouette score
    """
    sil_samples = silhouette_samples(embeddings, pred_labels)
    unique_labels = np.unique(pred_labels)
    per_class_scores = []

    for c in unique_labels:
        class_mask = (pred_labels == c)
        if np.sum(class_mask) > 1:  # need at least 2 samples
            per_class_scores.append(sil_samples[class_mask].mean())            
    return np.mean(per_class_scores) if per_class_scores else 0


class UnsupervisedMetricsTracker:
    """Track and visualize metrics during unsupervised training."""
    
    def __init__(self, model, ckpt_dir, num_labels, args, device):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.num_labels = num_labels
        self.args = args
        self.device = device
        self.silhouette_scores = []
        self.best_silhouette = -1
        self.best_epoch = -1
        self.patience = 10
        self.plateau_counter = 0
        self.min_delta = 0.005
        self.epoch_stats = []

    def save_epoch_stats_to_csv(self):
        """Save epoch statistics to CSV file."""
        save_dir = self.args.checkpoint_dir if hasattr(self.args, 'checkpoint_dir') and self.args.checkpoint_dir else self.ckpt_dir
        df = pd.DataFrame(self.epoch_stats, columns=['epoch', 'loss', 'macro_silhouette'])
        df.to_csv(os.path.join(save_dir, 'epoch_stats.csv'), index=False)

    def create_tsne_plot(self, epoch, embeddings, cluster_labels):
        """Create and save t-SNE visualization."""
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_result = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            indices = cluster_labels == label
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                        c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        plt.title(f't-SNE Visualization - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.ckpt_dir, f'tsne_epoch_{epoch}.png'))
        plt.close()

    def evaluate_epoch(self, epoch, dataloader, loss, logger):
        """
        Evaluate model at the end of an epoch.
        
        Args:
            epoch: Current epoch number
            dataloader: Validation dataloader
            loss: Training loss for this epoch
            logger: Logger instance
        
        Returns:
            Tuple of (is_best, early_stop, macro_sil)
        """
        self.model.eval()
        all_embeddings = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, val_labels = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids, attention_mask)
                embeddings = outputs[0][:, 0].cpu().numpy()  # CLS token embedding
                all_embeddings.append(embeddings)
                val_labels_list.append(val_labels.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        val_labels = np.concatenate(val_labels_list)

        macro_sil = macro_sil_score(all_embeddings, val_labels) if len(np.unique(val_labels)) > 1 else 0
        micro_sil = silhouette_score(all_embeddings, val_labels) if len(np.unique(val_labels)) > 1 else 0
        
        self.silhouette_scores.append(macro_sil)
        logger.info(f"Epoch {epoch} - Macro Silhouette Score: {macro_sil:.4f}")
        logger.info(f"Epoch {epoch} - Micro Silhouette Score: {micro_sil:.4f}")

        self.epoch_stats.append({
            'epoch': epoch, 
            'loss': loss, 
            'macro_silhouette': macro_sil,                            
        })

        # Create plot
        if self.ckpt_dir and not self.args.tune:
            self.create_tsne_plot(epoch, all_embeddings, val_labels)

        # Check for best model
        is_best = False
        early_stop = False
        
        if macro_sil > self.best_silhouette:
            self.best_silhouette = macro_sil
            self.best_epoch = epoch
            self.plateau_counter = 0
            is_best = True
        else:
            if macro_sil <= self.best_silhouette - self.min_delta:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
                
            if self.plateau_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                early_stop = True
        
        return is_best, early_stop, macro_sil