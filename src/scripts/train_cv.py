#!/usr/bin/env python3
"""
TAPESTRY: Cross-Validation Training for Supervised TF Estimator

Implements k-fold cross-validation with patient-level splitting to get
unbiased performance estimates on all samples.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class SupervisedTFEstimator(nn.Module):
    """Simple classifier + regressor for TF estimation"""
    def __init__(self, n_regions=5000, hidden_dim=512):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(n_regions * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
            # No sigmoid - use BCEWithLogitsLoss
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, methylation_counts, coverage):
        # Compute beta values
        beta = methylation_counts / (coverage + 1e-6)

        # Concatenate features
        x = torch.cat([beta, torch.log1p(coverage)], dim=-1)

        # Extract features
        features = self.features(x)

        # Two heads
        cancer_logits = self.classifier(features).squeeze(-1)
        tumor_fraction = self.regressor(features).squeeze(-1)

        return cancer_logits, tumor_fraction


def patient_stratified_kfold(metadata, n_splits=5):
    """
    Create patient-level folds, stratified by cancer/healthy

    Returns:
        list of (train_idx, val_idx) tuples
    """
    logger = logging.getLogger(__name__)

    # Get unique patients with their labels
    logger.info("Creating patient-level stratified folds...")
    patient_labels = metadata.groupby('patient_id')['is_cancer'].first()
    unique_patients = patient_labels.index.values
    labels = patient_labels.values

    logger.info(f"  Total unique patients: {len(unique_patients)}")
    logger.info(f"  Cancer patients: {labels.sum()}")
    logger.info(f"  Healthy patients: {(labels == 0).sum()}")

    # Stratified split by patient
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []
    for fold_num, (train_patient_idx, val_patient_idx) in enumerate(skf.split(unique_patients, labels), 1):
        train_patients = unique_patients[train_patient_idx]
        val_patients = unique_patients[val_patient_idx]

        # Get all sample indices for these patients
        train_idx = metadata[metadata['patient_id'].isin(train_patients)].index.values
        val_idx = metadata[metadata['patient_id'].isin(val_patients)].index.values

        train_cancer = metadata.loc[train_idx, 'is_cancer'].sum()
        val_cancer = metadata.loc[val_idx, 'is_cancer'].sum()

        folds.append((train_idx, val_idx))
        logger.info(f"  Fold {fold_num}: {len(train_patients)} train patients ({len(train_idx)} samples, {train_cancer} cancer), "
                   f"{len(val_patients)} val patients ({len(val_idx)} samples, {val_cancer} cancer)")

    return folds


def train_one_fold(
    train_meth, train_cov, train_labels, train_tf,
    val_meth, val_cov, val_labels, val_tf,
    fold_num, n_epochs, learning_rate, device, logger
):
    """
    Train model on one fold with per-epoch validation metrics

    Returns:
        val_cancer_prob: predictions on validation fold
        val_tf_pred: TF predictions on validation fold
        history: training history with validation metrics per epoch
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold_num}")
    logger.info(f"{'='*60}")

    # Convert to tensors
    logger.info("Moving data to device...")
    X_train = torch.FloatTensor(train_meth).to(device)
    C_train = torch.FloatTensor(train_cov).to(device)
    y_train = torch.FloatTensor(train_labels).to(device)
    tf_train = torch.FloatTensor(train_tf).to(device)

    X_val = torch.FloatTensor(val_meth).to(device)
    C_val = torch.FloatTensor(val_cov).to(device)
    y_val = torch.FloatTensor(val_labels).to(device)
    tf_val = torch.FloatTensor(val_tf).to(device)

    # Reliable mask (for regression supervision)
    reliable_train = (y_train == 1) & (tf_train > 0.05)
    reliable_val = (y_val == 1) & (tf_val > 0.05)

    logger.info(f"  Train: {len(train_labels)} samples ({(y_train==1).sum().item()} cancer, {reliable_train.sum().item()} reliable TF)")
    logger.info(f"  Val: {len(val_labels)} samples ({(y_val==1).sum().item()} cancer, {reliable_val.sum().item()} reliable TF)")

    # Initialize model
    logger.info("Initializing model...")
    model = SupervisedTFEstimator(n_regions=train_meth.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False
    )

    # Training history
    history = {
        'epoch': [],
        'train_loss': [], 'train_cls_loss': [], 'train_reg_loss': [],
        'val_loss': [], 'val_cls_loss': [], 'val_reg_loss': [],
        'val_acc': [], 'val_auc': [],
        'val_tf_corr_all': [], 'val_tf_mae_all': [],
        'val_tf_corr_high': [], 'val_tf_mae_high': [],
        'lr': []
    }

    logger.info(f"Starting training for {n_epochs} epochs...")
    logger.info(f"{'='*60}")

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30

    for epoch in range(n_epochs):
        # ============================================================
        # TRAINING
        # ============================================================
        model.train()
        optimizer.zero_grad()

        # Forward pass
        cancer_logits, tf_pred = model(X_train, C_train)

        # Loss 1: Binary classification
        cls_loss = F.binary_cross_entropy_with_logits(cancer_logits, y_train)

        # Loss 2: TF regression (reliable cancer only)
        if reliable_train.sum() > 0:
            reg_loss = F.mse_loss(tf_pred[reliable_train], tf_train[reliable_train])
        else:
            reg_loss = torch.tensor(0.0).to(device)

        # Loss 3: Healthy should predict TF ~0
        healthy_mask = y_train == 0
        if healthy_mask.sum() > 0:
            healthy_tf_loss = torch.mean(tf_pred[healthy_mask] ** 2)
        else:
            healthy_tf_loss = torch.tensor(0.0).to(device)

        # Combined loss (gradually increase regression weight)
        reg_weight = min(50.0, 5.0 + epoch * 0.5)
        train_loss = cls_loss + reg_weight * reg_loss + 10.0 * healthy_tf_loss

        # Backward pass
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ============================================================
        # VALIDATION (every epoch)
        # ============================================================
        model.eval()
        with torch.no_grad():
            # Forward pass on validation
            val_cancer_logits, val_tf_pred = model(X_val, C_val)

            # Validation losses
            val_cls_loss = F.binary_cross_entropy_with_logits(val_cancer_logits, y_val)

            if reliable_val.sum() > 0:
                val_reg_loss = F.mse_loss(val_tf_pred[reliable_val], tf_val[reliable_val])
            else:
                val_reg_loss = torch.tensor(0.0).to(device)

            val_healthy_mask = y_val == 0
            if val_healthy_mask.sum() > 0:
                val_healthy_tf_loss = torch.mean(val_tf_pred[val_healthy_mask] ** 2)
            else:
                val_healthy_tf_loss = torch.tensor(0.0).to(device)

            val_loss = val_cls_loss + reg_weight * val_reg_loss + 10.0 * val_healthy_tf_loss

            # Classification metrics
            val_cancer_prob = torch.sigmoid(val_cancer_logits).cpu().numpy()
            val_labels_np = y_val.cpu().numpy()

            val_acc = ((val_cancer_prob > 0.5) == val_labels_np).mean()
            if len(np.unique(val_labels_np)) > 1:
                val_auc = roc_auc_score(val_labels_np, val_cancer_prob)
            else:
                val_auc = 0.0

            # Regression metrics (all cancer samples)
            val_tf_pred_np = val_tf_pred.cpu().numpy()
            val_tf_np = tf_val.cpu().numpy()

            cancer_mask = val_labels_np == 1
            if cancer_mask.sum() > 0:
                cancer_tf_true = val_tf_np[cancer_mask]
                cancer_tf_pred = val_tf_pred_np[cancer_mask]

                if len(cancer_tf_true) > 1:
                    corr_all, _ = pearsonr(cancer_tf_pred, cancer_tf_true)
                else:
                    corr_all = 0.0
                mae_all = np.abs(cancer_tf_pred - cancer_tf_true).mean()

                # High TF only
                high_tf_mask = cancer_tf_true > 0.05
                if high_tf_mask.sum() > 1:
                    corr_high, _ = pearsonr(
                        cancer_tf_pred[high_tf_mask],
                        cancer_tf_true[high_tf_mask]
                    )
                    mae_high = np.abs(
                        cancer_tf_pred[high_tf_mask] - cancer_tf_true[high_tf_mask]
                    ).mean()
                else:
                    corr_high = 0.0
                    mae_high = 0.0
            else:
                corr_all = mae_all = corr_high = mae_high = 0.0

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.item())
        history['train_cls_loss'].append(cls_loss.item())
        history['train_reg_loss'].append(reg_loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_cls_loss'].append(val_cls_loss.item())
        history['val_reg_loss'].append(val_reg_loss.item())
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_tf_corr_all'].append(corr_all)
        history['val_tf_mae_all'].append(mae_all)
        history['val_tf_corr_high'].append(corr_high)
        history['val_tf_mae_high'].append(mae_high)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        scheduler.step(val_loss)

        # Logging (every 10 epochs or if best)
        is_best = val_loss.item() < best_val_loss
        if epoch % 10 == 0 or is_best:
            logger.info(
                f"Epoch {epoch:4d} | "
                f"Train: {train_loss.item():.4f} (cls:{cls_loss.item():.4f} reg:{reg_loss.item():.4f}) | "
                f"Val: {val_loss.item():.4f} (cls:{val_cls_loss.item():.4f} reg:{val_reg_loss.item():.4f}) | "
                f"Acc:{val_acc:.3f} AUC:{val_auc:.3f} | "
                f"TF_corr:{corr_all:.3f} MAE:{mae_all:.4f} | "
                f"TF_high_corr:{corr_high:.3f} MAE_high:{mae_high:.4f}"
                + (" *BEST*" if is_best else "")
            )

        # Early stopping check
        if is_best:
            best_val_loss = val_loss.item()
            patience_counter = 0
            # Save best model predictions
            best_val_cancer_prob = val_cancer_prob.copy()
            best_val_tf_pred = val_tf_pred_np.copy()
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
            break

    logger.info(f"{'='*60}")
    logger.info(f"Fold {fold_num} training complete!")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}\n")

    # Return predictions from best model
    return best_val_cancer_prob, best_val_tf_pred, history


def cross_validate(
    meth_data, cov_data, metadata,
    n_splits=5, n_epochs=200, learning_rate=1e-4, device='cuda'
):
    """
    Run k-fold cross-validation

    Returns:
        all_cancer_probs: cancer probabilities for all samples
        all_tf_preds: TF predictions for all samples
        fold_histories: training history for each fold
        fold_assigned: which fold each sample was in
    """
    logger = logging.getLogger(__name__)

    n_samples = meth_data.shape[0]

    # Initialize arrays
    all_cancer_probs = np.zeros(n_samples)
    all_tf_preds = np.zeros(n_samples)
    fold_assigned = np.zeros(n_samples, dtype=int)
    fold_histories = []

    # Create patient-level folds
    folds = patient_stratified_kfold(metadata, n_splits=n_splits)

    # Train each fold
    for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
        # Extract data
        train_meth = meth_data[train_idx]
        train_cov = cov_data[train_idx]
        train_labels = metadata.loc[train_idx, 'is_cancer'].values
        train_tf = metadata.loc[train_idx, 'tumor_fraction'].values

        val_meth = meth_data[val_idx]
        val_cov = cov_data[val_idx]
        val_labels = metadata.loc[val_idx, 'is_cancer'].values
        val_tf = metadata.loc[val_idx, 'tumor_fraction'].values

        # Train and predict
        val_cancer_prob, val_tf_pred, history = train_one_fold(
            train_meth, train_cov, train_labels, train_tf,
            val_meth, val_cov, val_labels, val_tf,
            fold_num=fold_num,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device=device,
            logger=logger
        )

        # Store results
        all_cancer_probs[val_idx] = val_cancer_prob
        all_tf_preds[val_idx] = val_tf_pred
        fold_assigned[val_idx] = fold_num
        fold_histories.append(history)

    return all_cancer_probs, all_tf_preds, fold_assigned, fold_histories


def evaluate_cv_results(metadata, cancer_probs, tf_preds):
    """Evaluate cross-validation results across all samples"""
    logger = logging.getLogger(__name__)

    true_labels = metadata['is_cancer'].values
    true_tf = metadata['tumor_fraction'].values

    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION RESULTS (All Samples)")
    logger.info("="*60)

    # Classification metrics
    auc = roc_auc_score(true_labels, cancer_probs)
    predictions = (cancer_probs > 0.5).astype(int)
    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    logger.info("\nCLASSIFICATION PERFORMANCE:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"                  Predicted")
    logger.info(f"                Healthy  Cancer")
    logger.info(f"  Actual Healthy  {cm[0,0]:4d}    {cm[0,1]:4d}")
    logger.info(f"         Cancer   {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Regression metrics (cancer samples only)
    cancer_mask = true_labels == 1
    if cancer_mask.sum() > 0:
        cancer_tf_true = true_tf[cancer_mask]
        cancer_tf_pred = tf_preds[cancer_mask]

        # All cancer samples
        corr_all, pval_all = pearsonr(cancer_tf_pred, cancer_tf_true)
        mae_all = np.abs(cancer_tf_pred - cancer_tf_true).mean()

        # High TF only (> 5%)
        high_tf_mask = cancer_tf_true > 0.05
        if high_tf_mask.sum() > 1:
            corr_high, pval_high = pearsonr(
                cancer_tf_pred[high_tf_mask],
                cancer_tf_true[high_tf_mask]
            )
            mae_high = np.abs(
                cancer_tf_pred[high_tf_mask] - cancer_tf_true[high_tf_mask]
            ).mean()
        else:
            corr_high = pval_high = mae_high = 0

        logger.info("\nTUMOR FRACTION PREDICTION (Cancer samples):")
        logger.info(f"  All cancer samples (n={cancer_mask.sum()}):")
        logger.info(f"    Correlation: {corr_all:.4f} (p={pval_all:.4e})")
        logger.info(f"    MAE: {mae_all:.4f}")
        logger.info(f"\n  High TF (>5%) samples (n={high_tf_mask.sum()}):")
        logger.info(f"    Correlation: {corr_high:.4f} (p={pval_high:.4e})")
        logger.info(f"    MAE: {mae_high:.4f}")

    # Healthy samples check
    healthy_mask = true_labels == 0
    if healthy_mask.sum() > 0:
        healthy_tf_pred = tf_preds[healthy_mask]
        false_positives = (healthy_tf_pred > 0.01).sum()

        logger.info("\nHEALTHY CONTROLS CHECK:")
        logger.info(f"  Mean predicted TF: {healthy_tf_pred.mean():.5f}")
        logger.info(f"  Max predicted TF: {healthy_tf_pred.max():.5f}")
        logger.info(f"  False positives (>1%): {false_positives}/{healthy_mask.sum()}")

    logger.info("="*60)

    return {
        'auc': auc,
        'accuracy': acc,
        'confusion_matrix': cm,
        'tf_correlation_all': corr_all if cancer_mask.sum() > 0 else None,
        'tf_correlation_high': corr_high if cancer_mask.sum() > 0 and high_tf_mask.sum() > 0 else None,
        'mae_all': mae_all if cancer_mask.sum() > 0 else None,
        'mae_high': mae_high if cancer_mask.sum() > 0 and high_tf_mask.sum() > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Cross-validation training for supervised TF estimator'
    )
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Directory with methylation_matrix.npy, coverage_matrix.npy')
    parser.add_argument('--sample-ids', type=Path, required=True,
                       help='CSV with sample_ids')
    parser.add_argument('--ichorcna-file', type=Path, required=True,
                       help='CSV with columns: sample_name, ichorCNA_tf')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Max epochs per fold')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Setup logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(args.output_dir / 'cv_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("TAPESTRY: Cross-Validation Training")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    logger.info(f"  Methylation matrix from: {args.data_dir / 'methylation_matrix.npy'}")
    meth_data = np.load(args.data_dir / 'methylation_matrix.npy')
    logger.info(f"    Shape: {meth_data.shape}")

    logger.info(f"  Coverage matrix from: {args.data_dir / 'coverage_matrix.npy'}")
    cov_data = np.load(args.data_dir / 'coverage_matrix.npy')
    logger.info(f"    Shape: {cov_data.shape}")

    logger.info(f"  Sample IDs from: {args.sample_ids}")
    sample_ids_df = pd.read_csv(args.sample_ids)
    logger.info(f"    Total samples: {len(sample_ids_df)}")

    logger.info(f"  ichorCNA data from: {args.ichorcna_file}")
    ichorcna = pd.read_csv(args.ichorcna_file)
    logger.info(f"    Total records: {len(ichorcna)}")

    # Prepare metadata
    logger.info("\nPreparing metadata...")
    ichorcna = ichorcna.rename(columns={'sample_name': 'sample_id', 'TF': 'ichorCNA_tf'})

    # Extract patient_id from sample_id (assuming format like "GI001_TP1" or "SCAN002_Bsl")
    sample_ids_df['patient_id'] = sample_ids_df['sample_id'].str.extract(r'([A-Z]+\d+)')[0]

    metadata = sample_ids_df.merge(
        ichorcna[['sample_id', 'ichorCNA_tf']],
        on='sample_id',
        how='left'
    )

    # Create labels
    metadata['is_cancer'] = (metadata['ichorCNA_tf'] > 0).astype(int)
    metadata['tumor_fraction'] = metadata['ichorCNA_tf'].fillna(0)

    logger.info(f"\nDataset summary:")
    logger.info(f"  Total samples: {len(metadata)}")
    logger.info(f"  Unique patients: {metadata['patient_id'].nunique()}")
    logger.info(f"  Healthy (TF=0): {(metadata['is_cancer'] == 0).sum()}")
    logger.info(f"  Cancer (TF>0): {(metadata['is_cancer'] == 1).sum()}")
    logger.info(f"  High TF (>5%): {(metadata['tumor_fraction'] > 0.05).sum()}")
    logger.info(f"  Features: {meth_data.shape[1]:,} regions")

    # Run cross-validation
    logger.info(f"\nStarting {args.n_folds}-fold cross-validation...")
    cancer_probs, tf_preds, fold_assigned, fold_histories = cross_validate(
        meth_data=meth_data,
        cov_data=cov_data,
        metadata=metadata,
        n_splits=args.n_folds,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

    # Add predictions to metadata
    metadata['cancer_prob_cv'] = cancer_probs
    metadata['tf_pred_cv'] = tf_preds
    metadata['cv_fold'] = fold_assigned

    # Evaluate overall performance
    results = evaluate_cv_results(metadata, cancer_probs, tf_preds)

    # Save results
    logger.info("\nSaving results...")
    metadata.to_csv(args.output_dir / 'cv_predictions.csv', index=False)
    logger.info(f"  Predictions saved to: {args.output_dir / 'cv_predictions.csv'}")

    # Save fold histories
    for fold_num, history in enumerate(fold_histories, 1):
        history_df = pd.DataFrame(history)
        history_df.to_csv(args.output_dir / f'fold_{fold_num}_history.csv', index=False)
    logger.info(f"  Training histories saved to: {args.output_dir}/fold_*_history.csv")

    # Save summary
    with open(args.output_dir / 'cv_summary.txt', 'w') as f:
        f.write("TAPESTRY Cross-Validation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Folds: {args.n_folds}\n")
        f.write(f"  Max epochs per fold: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Device: {args.device}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Classification AUC: {results['auc']:.4f}\n")
        f.write(f"  Classification Accuracy: {results['accuracy']:.4f}\n")
        if results['tf_correlation_all'] is not None:
            f.write(f"  TF Correlation (all): {results['tf_correlation_all']:.4f}\n")
            f.write(f"  TF MAE (all): {results['mae_all']:.4f}\n")
        if results['tf_correlation_high'] is not None:
            f.write(f"  TF Correlation (TF>5%): {results['tf_correlation_high']:.4f}\n")
            f.write(f"  TF MAE (TF>5%): {results['mae_high']:.4f}\n")

    logger.info(f"  Summary saved to: {args.output_dir / 'cv_summary.txt'}")
    logger.info("\n" + "="*60)
    logger.info("Cross-validation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
