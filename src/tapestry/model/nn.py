import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedTFEstimator(nn.Module):
    """
    Direct supervised learning: methylation → (is_cancer, tumor_fraction)
    """
    def __init__(self, n_regions=5000, hidden_dim=512):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(n_regions * 2, hidden_dim),  # *2 for methylation + coverage
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
        
        # Classification head (healthy vs cancer)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
            # No sigmoid here - use BCEWithLogitsLoss for numerical stability
        )
        
        # Regression head (tumor fraction)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # TF in [0, 1]
        )
    
    def forward(self, methylation_counts, coverage):
        """
        Args:
            methylation_counts: [batch_size, n_regions]
            coverage: [batch_size, n_regions]
            
        Returns:
            cancer_logits: [batch_size] - raw logits for classification
            tumor_fraction: [batch_size] - predicted TF in [0, 1]
        """
        # Compute beta values (methylation rate)
        beta = methylation_counts / (coverage + 1e-6)
        
        # Concatenate features: [beta values, log(coverage)]
        x = torch.cat([beta, torch.log1p(coverage)], dim=-1)  # [batch, n_regions*2]
        
        # Extract features
        features = self.features(x)  # [batch, hidden_dim/4]
        
        # Two heads
        cancer_logits = self.classifier(features).squeeze(-1)  # [batch]
        tumor_fraction = self.regressor(features).squeeze(-1)  # [batch]
        
        return cancer_logits, tumor_fraction
    
    def predict(self, methylation_counts, coverage):
        """
        Convenience method for inference
        """
        self.eval()
        with torch.no_grad():
            cancer_logits, tumor_fraction = self.forward(methylation_counts, coverage)
            cancer_prob = torch.sigmoid(cancer_logits)
            
        return {
            'cancer_probability': cancer_prob,
            'tumor_fraction': tumor_fraction,
            'prediction': (cancer_prob > 0.5).long()  # Binary prediction
        }
    
def train_supervised_model(
    methylation_counts,  # [n_samples, n_regions]
    coverage,            # [n_samples, n_regions]
    labels,              # [n_samples] - 0=healthy, 1=cancer
    tumor_fraction,      # [n_samples] - ichorCNA TF
    is_reliable,         # [n_samples] - bool, True if TF > 0.05
    n_epochs=200,
    learning_rate=1e-4,
    device='cuda'
):
    """
    Train the supervised model
    """
    print(f"\nMoving data to device: {device}...")
    # Move data to device
    meth = torch.FloatTensor(methylation_counts).to(device)
    cov = torch.FloatTensor(coverage).to(device)
    labels_t = torch.FloatTensor(labels).to(device)
    tf = torch.FloatTensor(tumor_fraction).to(device)
    reliable = torch.BoolTensor(is_reliable).to(device)
    print("  Done.")

    # Initialize model
    print(f"Initializing model with {meth.shape[1]} regions...")
    model = SupervisedTFEstimator(n_regions=meth.shape[1]).to(device)
    print("  Done.")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )
    
    # Training loop
    print("\nStarting training loop...")
    history = {'train_loss': [], 'cls_loss': [], 'reg_loss': []}

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        cancer_logits, tf_pred = model(meth, cov)
        
        # Loss 1: Binary classification (all samples)
        cls_loss = F.binary_cross_entropy_with_logits(
            cancer_logits,
            labels_t
        )
        
        # Loss 2: TF regression (only reliable cancer samples)
        cancer_mask = labels_t == 1
        reliable_cancer = cancer_mask & reliable
        
        if reliable_cancer.sum() > 0:
            reg_loss = F.mse_loss(
                tf_pred[reliable_cancer],
                tf[reliable_cancer]
            )
        else:
            reg_loss = torch.tensor(0.0).to(device)
        
        # Loss 3: Healthy samples should predict TF ~0
        healthy_mask = labels_t == 0
        if healthy_mask.sum() > 0:
            healthy_tf_loss = torch.mean(tf_pred[healthy_mask] ** 2)  # Push to 0
        else:
            healthy_tf_loss = torch.tensor(0.0).to(device)
        
        # Combined loss
        # Weight regression higher once classification is working
        reg_weight = min(50.0, 5.0 + epoch * 0.5)  # Gradually increase
        
        total_loss = (
            cls_loss +                    # Classification (always important)
            reg_weight * reg_loss +       # TF regression (increase over time)
            10.0 * healthy_tf_loss        # Healthy should be ~0 TF
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        history['train_loss'].append(total_loss.item())
        history['cls_loss'].append(cls_loss.item())
        history['reg_loss'].append(reg_loss.item())
        
        scheduler.step(total_loss)
        
        # Logging
        if epoch % 10 == 0:
            with torch.no_grad():
                cancer_prob = torch.sigmoid(cancer_logits)
                acc = ((cancer_prob > 0.5).float() == labels_t).float().mean()
                
                if reliable_cancer.sum() > 0:
                    mae = torch.abs(tf_pred[reliable_cancer] - tf[reliable_cancer]).mean()
                    corr = torch.corrcoef(
                        torch.stack([tf_pred[reliable_cancer], tf[reliable_cancer]])
                    )[0, 1]
                else:
                    mae = 0.0
                    corr = 0.0
            
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | "
                  f"Cls: {cls_loss:.4f} | Reg: {reg_loss:.4f} | "
                  f"Acc: {acc:.3f} | MAE: {mae:.4f} | Corr: {corr:.3f}")
    
    return model, history


def evaluate_model(model, methylation_counts, coverage, labels, tumor_fraction, device='cuda'):
    """
    Evaluate model performance
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    from scipy.stats import pearsonr
    
    model.eval()
    
    # Move data to device
    meth = torch.FloatTensor(methylation_counts).to(device)
    cov = torch.FloatTensor(coverage).to(device)
    
    # Predict
    with torch.no_grad():
        cancer_logits, tf_pred = model(meth, cov)
        cancer_prob = torch.sigmoid(cancer_logits)
    
    # Move to numpy
    cancer_prob_np = cancer_prob.cpu().numpy()
    tf_pred_np = tf_pred.cpu().numpy()
    
    # Classification metrics
    auc = roc_auc_score(labels, cancer_prob_np)
    predictions = (cancer_prob_np > 0.5).astype(int)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    print("\n" + "="*60)
    print("CLASSIFICATION PERFORMANCE (Healthy vs Cancer)")
    print("="*60)
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Healthy  Cancer")
    print(f"Actual Healthy  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Cancer   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Regression metrics (only for cancer samples)
    cancer_mask = labels == 1
    if cancer_mask.sum() > 0:
        cancer_tf_true = tumor_fraction[cancer_mask]
        cancer_tf_pred = tf_pred_np[cancer_mask]
        
        # Overall correlation
        corr_all, pval_all = pearsonr(cancer_tf_pred, cancer_tf_true)
        mae_all = np.abs(cancer_tf_pred - cancer_tf_true).mean()
        
        # High TF only (> 5%)
        high_tf_mask = cancer_tf_true > 0.05
        if high_tf_mask.sum() > 0:
            corr_high, pval_high = pearsonr(
                cancer_tf_pred[high_tf_mask],
                cancer_tf_true[high_tf_mask]
            )
            mae_high = np.abs(
                cancer_tf_pred[high_tf_mask] - cancer_tf_true[high_tf_mask]
            ).mean()
        else:
            corr_high, pval_high, mae_high = 0, 1, 0
        
        print("\n" + "="*60)
        print("TUMOR FRACTION PREDICTION (Cancer samples only)")
        print("="*60)
        print(f"All cancer samples:")
        print(f"  Correlation: {corr_all:.4f} (p={pval_all:.4e})")
        print(f"  MAE: {mae_all:.4f}")
        print(f"\nHigh TF (>5%) samples:")
        print(f"  Correlation: {corr_high:.4f} (p={pval_high:.4e})")
        print(f"  MAE: {mae_high:.4f}")
    
    # Healthy samples check
    healthy_mask = labels == 0
    if healthy_mask.sum() > 0:
        healthy_tf_pred = tf_pred_np[healthy_mask]
        false_positives = (healthy_tf_pred > 0.01).sum()
        
        print("\n" + "="*60)
        print("HEALTHY CONTROLS CHECK")
        print("="*60)
        print(f"Mean predicted TF: {healthy_tf_pred.mean():.5f}")
        print(f"Max predicted TF: {healthy_tf_pred.max():.5f}")
        print(f"False positives (>1%): {false_positives}/{healthy_mask.sum()}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'confusion_matrix': cm,
        'tf_correlation_all': corr_all if cancer_mask.sum() > 0 else None,
        'tf_correlation_high': corr_high if cancer_mask.sum() > 0 and high_tf_mask.sum() > 0 else None,
        'mae_all': mae_all if cancer_mask.sum() > 0 else None,
        'mae_high': mae_high if cancer_mask.sum() > 0 and high_tf_mask.sum() > 0 else None,
    }


if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Train supervised TF estimator')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with methylation_matrix.npy, coverage_matrix.npy, sample_ids.csv')
    parser.add_argument('--ichorcna-file', type=Path, required=True,
                        help='CSV with columns: sample_name, ichorCNA_tf')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()

    print("="*60)
    print("SUPERVISED TF ESTIMATOR - Training")
    print("="*60)

    # Load data
    print("\nLoading methylation matrix...")
    methylation_counts = np.load(args.data_dir / 'methylation_matrix.npy')
    print(f"  Shape: {methylation_counts.shape}")

    print("Loading coverage matrix...")
    coverage = np.load(args.data_dir / 'coverage_matrix.npy')
    print(f"  Shape: {coverage.shape}")

    print("Loading sample IDs...")
    sample_ids_df = pd.read_csv(args.data_dir / 'sample_ids.csv')
    sample_ids = sample_ids_df['sample_id'].tolist()
    print(f"  Total samples: {len(sample_ids)}")

    # Load ichorCNA TF
    print("Loading ichorCNA tumor fractions...")
    ichorcna = pd.read_csv(args.ichorcna_file)
    print(f"  Total records in ichorCNA file: {len(ichorcna)}")
    ichorcna = ichorcna.rename(columns={'sample_name': 'sample_id', 'TF': 'ichorCNA_tf'})

    # Merge with sample IDs
    print("Merging metadata...")
    metadata = pd.DataFrame({'sample_id': sample_ids})
    metadata = metadata.merge(ichorcna[['sample_id', 'ichorCNA_tf']], on='sample_id', how='left')

    # Create labels: 0=healthy (TF==0), 1=cancer (TF>0)
    print("Creating labels...")
    labels = (metadata['ichorCNA_tf'] > 0).astype(int).values
    tumor_fraction = metadata['ichorCNA_tf'].fillna(0).values
    is_reliable = (tumor_fraction > 0.05)

    print(f"\nDataset summary:")
    print(f"  Total samples: {len(sample_ids)}")
    print(f"  Healthy (TF=0): {(labels == 0).sum()}")
    print(f"  Cancer (TF>0): {(labels == 1).sum()}")
    print(f"  Reliable TF (>5%): {is_reliable.sum()}")
    print(f"  Features: {methylation_counts.shape[1]:,} regions")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("="*60)

    model, history = train_supervised_model(
        methylation_counts=methylation_counts,
        coverage=coverage,
        labels=labels,
        tumor_fraction=tumor_fraction,
        is_reliable=is_reliable,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model,
        methylation_counts,
        coverage,
        labels,
        tumor_fraction,
        device=args.device
    )

    # Save model
    model_path = args.output_dir / 'supervised_tf_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_regions': methylation_counts.shape[1],
        'history': history,
        'results': results,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(args.output_dir / 'training_history.csv', index=False)
    print(f"History saved to: {args.output_dir / 'training_history.csv'}")