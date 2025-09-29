"""
Training and evaluation functions.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from losses import predict_lpa, supervised_contrastive_loss


batch_log = list()


def train_model(args, model, train_loader, val_loader, optimizer, class_weights, num_labels, ckpt_dir, logger, device):
    """Train the BertClassifier model."""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device)).to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    train_loss_list = []
    val_f1_list = []

    global batch_log
    batch_log.clear()
    sigma = args.sigma if args.sigma is not None else 0.1
    lambda_param = args.lam if args.lam is not None else 0.1
    gamma = args.gamma if args.gamma is not None else 0.7
    epoch_stats, epoch_times  = [], []
    gloss_stats, ce_loss_stats, scl_stats, combined_stats = [], [], [], []
    
    for epoch in range(args.nb_epochs):
        model.train()
        t0_epoch = time.time()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            ce_loss = criterion(logits, labels)
            ce_loss_stats.append(ce_loss.item())
            
            if args.loss == 'gloss':
                gloss = predict_lpa(model, input_ids, attention_mask, labels, sigma, num_labels, gamma, device, class_weights, logger)
                gloss_stats.append(gloss.item())
                loss = (1-lambda_param) * ce_loss + lambda_param * gloss
                combined_stats.append(loss.item())
            elif args.loss == 'scl':
                lambda_param = 0.9
                scl_loss = supervised_contrastive_loss(model, input_ids, attention_mask, labels, args.temperature)
                scl_stats.append(scl_loss.item())
                loss = (1-lambda_param) * ce_loss + lambda_param * scl_loss
                combined_stats.append(loss.item())
            else:
                loss = ce_loss
                
            loss.backward()        
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        epoch_time = time.time() - t0_epoch
        epoch_times.append(epoch_time)

        # Evaluate on Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1_list.append(val_f1)
        
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{args.nb_epochs}, Train Loss: {avg_train_loss:.4f}, Val F1: {val_f1*100:.2f}, Time: {epoch_time:.2f}s")
        epoch_stats.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_f1': val_f1*100, 'time': epoch_time})

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(ckpt_dir, 'checkpoint.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    if not args.tune:
        # plot train loss
        plt.figure()
        plt.plot(list(range(len(train_loss_list))), train_loss_list, label="Training Loss")
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(ckpt_dir, 'train_loss.png'))
        plt.close()

        # plot validation F1
        plt.figure()
        plt.plot(range(len(val_f1_list)), val_f1_list, label="Validation F1")
        plt.title('Validation F1')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.savefig(os.path.join(ckpt_dir, 'val_f1.png'))
        plt.close()

    logger.info("Training complete!")
    logger.info(f"Total time: {sum(epoch_times):.2f}s")
    logger.info(f"Avg train time per epoch: {sum(epoch_times)/len(epoch_times) if len(epoch_times) > 0 else 0:.2f}")
    pd.DataFrame(epoch_stats).to_csv(os.path.join(ckpt_dir, 'epoch_stats.csv'), index=False)

    if args.loss == 'ce':
        pd.DataFrame((ce_loss_stats,)).to_csv(os.path.join(ckpt_dir, 'loss_stats.csv'), index=False)
    elif args.loss == 'gloss':
        pd.DataFrame((gloss_stats, ce_loss_stats, combined_stats)).to_csv(os.path.join(ckpt_dir, 'loss_stats.csv'), index=False)
    else:
        pd.DataFrame((scl_stats, ce_loss_stats, combined_stats)).to_csv(os.path.join(ckpt_dir, 'loss_stats.csv'), index=False)
    
    return model, best_val_f1


def test_evaluate(model, test_loader, logger, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    logger.info(f"Classification report:\n{classification_report(all_labels, all_preds)}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1 score: {test_f1:.4f}")