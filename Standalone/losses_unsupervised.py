"""
Loss functions for unsupervised training.
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any
from sentence_transformers.util import fullname
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction


EPS = 1e-10


def normalize_adj(adj):
    """Normalize adjacency matrix."""
    adj = adj.to_dense() if hasattr(adj, 'to_dense') else adj
    rowsum = torch.sum(adj, dim=1)
    rowsum = rowsum.to(torch.float64)       
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
    ret = torch.where(torch.isnan(ret), torch.tensor(EPS, device=ret.device, dtype=ret.dtype), ret)
    ret = torch.where(torch.isinf(ret), torch.tensor(EPS, device=ret.device, dtype=ret.dtype), ret)
    return ret


def guassian(emb, sigma):
    """Compute Gaussian kernel weights."""
    sq_dists = torch.cdist(emb, emb, p=2) ** 2 
    weight = torch.exp(-sq_dists / (2*(sigma**2)))
    weight = weight - torch.diag(torch.diag(weight))
    return weight


def modified_lpa(train_emb, test_emb, Ytrain, sigma, num_labels, device, labels_orig=None):
    """Modified Label Propagation Algorithm for unsupervised setting."""
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0]
    labels = torch.cat((Ytrain, torch.zeros(test_emb.shape[0], device=device)), dim=0)

    Y = torch.zeros((num_nodes, num_labels), dtype=torch.float64, device=device)
    for k in range(num_labels):
        Y[labels == k, k] = 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[:Ytrain.shape[0]] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask[Ytrain.shape[0]:Ytrain.shape[0]+test_emb.shape[0]] = 1

    emb = emb / emb.norm(dim=1, keepdim=True)
    
    adj = guassian(emb, sigma).to(device)
    adj = adj.to(torch.float64)
    adj = adj + adj.t()
    
    if (torch.sum(torch.isnan(adj))) or (torch.sum(torch.isinf(adj))):
        raise ValueError("NaN in adj after symmetrization")
    
    adj_norm = normalize_adj(adj)
    adj_norm = adj_norm.to_dense()
    
    if (torch.sum(torch.isnan(adj_norm))) or (torch.sum(torch.isinf(adj_norm))):
        raise ValueError("NaN in adj after processing")

    # Transition matrix
    Tran = adj_norm / adj_norm.sum(dim=0, keepdim=True)
    row_sum = Tran.sum(dim=1, keepdim=True)
    T = Tran / row_sum

    N_l = train_emb.shape[0]
    T_ul = T[N_l:, :N_l]
    T_uu = T[N_l:, N_l:]

    I = torch.eye(T_uu.shape[0], dtype=torch.float64).to(device)
    F_UU = torch.linalg.solve(I - T_uu, T_ul.mm(Y[train_mask]))
    
    if torch.any(torch.isnan(F_UU)) or torch.any(torch.isinf(F_UU)):
        raise ValueError("NaN in F_UU before normalization")

    return F_UU


def predict_lpa(model, input_ids_lpa, attention_mask_lpa, labels, sigma, num_labels, gamma, device, class_weights): 
    """Compute G-Loss using Label Propagation."""
    embedding = model(input_ids_lpa, attention_mask_lpa)[0][:, 0]    
    
    mask1 = torch.randperm(embedding.size(0)) < embedding.size(0) * gamma
    mask2 = ~mask1
    emb_lab_set = embedding[mask1]
    emb_eval_set = embedding[mask2]

    labels_lab_set = labels[mask1]
    labels_eval_set = labels[mask2]
    predicted_labels = modified_lpa(emb_lab_set, emb_eval_set, labels_lab_set, sigma, num_labels, device=device, labels_orig=labels)
    loss = F.cross_entropy(predicted_labels, labels_eval_set, weight=torch.tensor(class_weights, dtype=torch.float64).to(device))
    return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float64).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
        

def supervised_contrastive_loss(model, input_ids, attention_mask, labels, temperature):
    """Compute supervised contrastive loss."""
    outputs = model(input_ids, attention_mask)
    embeddings = outputs[0][:, 0]  # Extract CLS token embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    loss = SupConLoss(temperature=temperature)(embeddings.unsqueeze(1), labels)
    return loss


class BatchAllTripletLoss(nn.Module):
    """Batch All Triplet Loss from SBERT library."""
    
    def __init__(
        self,
        model,  
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ) -> None:
        super(BatchAllTripletLoss, self).__init__()
        self.model = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, input_ids: torch.Tensor=None, attention_mask: torch.Tensor=None, 
                labels: torch.Tensor=None, emb:torch.Tensor=None, eval_mode=False) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)[0][:, 0]
        batch_loss = self.batch_all_triplet_loss(labels, outputs)
        return batch_loss

    def batch_all_triplet_loss(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        pairwise_dist = self.distance_metric(embeddings)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin
        mask = BatchHardTripletLoss.get_triplet_mask(labels)  # Put to zero the invalid triplets
        triplet_loss = mask.float() * triplet_loss
        triplet_loss[triplet_loss < 0] = 0         # Remove negative losses (i.e. the easy triplets)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]  # Count number of positive triplets
        num_positive_triplets = valid_triplets.size(0)
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        return triplet_loss


class BatchCosineSimilarityLoss(nn.Module):
    """Custom Batch Cosine Similarity Loss from SBERT library."""
    
    def __init__(self, model,
                 loss_fct: nn.Module = nn.MSELoss(), 
                 cos_score_transformation=nn.Identity()):
        super(BatchCosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        embeddings = outputs[0][:, 0]  # [batch_size, hidden_size]
        batch_loss = self.cos_similarity_loss(labels, embeddings)
        return batch_loss
    
    def cos_similarity_loss(self, labels:Tensor, embeddings: Tensor) -> Tensor:
        i, j = torch.triu_indices(len(embeddings), len(embeddings), 1)
        train_pairs = torch.stack((embeddings[i], embeddings[j]), dim=0)  # Shape: [ 2, num_pairs,embedding_dim]
        train_labels = (labels[i] == labels[j]).float()          # Shape: [num_pairs]
        output = self.cos_score_transformation(torch.cosine_similarity(train_pairs[0], train_pairs[1]))
        loss = self.loss_fct(output, train_labels.float().view(-1))
        return loss
    
    def get_config_dict(self) -> dict[str, Any]:
        return {"loss_fct": fullname(self.loss_fct)}