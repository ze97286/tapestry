#!/usr/bin/env python3
"""
TAPESTRY Step 2: Train Model

Train blind deconvolution VAE on aggregated regional data.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.model.vae import BlindDeconvolutionVAE
from tapestry.model.losses import TAPESTRYLoss
from tapestry.train.trainer import TAPESTRYTrainer


def setup_logging(log_file='tapestry_train.log'):
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def create_data_loaders(
    methylation: np.ndarray,
    coverage: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32
):
    """
    Create PyTorch DataLoaders.
    
    Args:
        methylation: [n_samples × n_regions]
        coverage: [n_samples × n_regions]
        train_indices: Indices for training set
        val_indices: Indices for validation set
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader
    """
    # Training set
    train_meth = torch.FloatTensor(methylation[train_indices])
    train_cov = torch.FloatTensor(coverage[train_indices])
    train_idx = torch.arange(len(train_indices))
    
    train_dataset = TensorDataset(train_meth, train_cov, train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Validation set
    val_meth = torch.FloatTensor(methylation[val_indices])
    val_cov = torch.FloatTensor(coverage[val_indices])
    val_idx = torch.arange(len(val_indices))
    
    val_dataset = TensorDataset(val_meth, val_cov, val_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Train blind deconvolution model'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory with processed data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models'),
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/default_config.yaml'),
        help='Configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TAPESTRY - Model Training")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading processed data...")
    
    methylation = np.load(args.data_dir / 'methylation_matrix.npy')
    coverage = np.load(args.data_dir / 'coverage_matrix.npy')
    regions = pd.read_csv(args.data_dir / 'regions.csv')
    sample_ids = pd.read_csv(args.data_dir / 'sample_ids.csv')
    
    logger.info(f"Data shape: {methylation.shape}")
    logger.info(f"Samples: {len(sample_ids)}")
    logger.info(f"Regions: {len(regions)}")
    
    # Load metadata (if available)
    metadata_path = args.data_dir / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        logger.info("Loaded metadata")
        
        # Extract tumor fractions (if available)
        if 'ichorCNA_tf' in metadata.columns:
            true_tf = metadata['ichorCNA_tf'].values
            # Mask for reliable estimates (TF > threshold)
            tf_mask = true_tf > config['training']['min_tf_reliability']
            logger.info(f"Found {tf_mask.sum()} samples with reliable TF estimates")
        else:
            true_tf = None
            tf_mask = None
            logger.warning("No ichorCNA_tf in metadata - training without TF supervision")
    else:
        true_tf = None
        tf_mask = None
        metadata = None
        logger.warning("No metadata found - training fully unsupervised")
    
    # Train/val split (patient-level if patient_id available)
    logger.info("\nCreating train/validation split...")
    
    n_samples = len(sample_ids)
    
    if metadata is not None and 'patient_id' in metadata.columns:
        # Patient-level split
        unique_patients = metadata['patient_id'].unique()
        n_train_patients = int(len(unique_patients) * 0.85)
        
        np.random.seed(42)
        train_patients = np.random.choice(
            unique_patients, 
            n_train_patients, 
            replace=False
        )
        
        train_mask = metadata['patient_id'].isin(train_patients)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(~train_mask)[0]
        
        logger.info(f"Train: {len(train_indices)} samples from {n_train_patients} patients")
        logger.info(f"Val: {len(val_indices)} samples from {len(unique_patients) - n_train_patients} patients")
    else:
        # Random split
        n_train = int(n_samples * 0.85)
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        logger.info(f"Train: {len(train_indices)} samples")
        logger.info(f"Val: {len(val_indices)} samples")
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    
    train_loader, val_loader = create_data_loaders(
        methylation,
        coverage,
        train_indices,
        val_indices,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    logger.info("\nInitializing model...")
    
    model = BlindDeconvolutionVAE(
        n_regions=methylation.shape[1],
        n_components=config['model']['n_components'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"  Components: {config['model']['n_components']}")
    logger.info(f"  Latent dim: {config['model']['latent_dim']}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss function
    loss_fn = TAPESTRYLoss(
        kl_weight=config['training']['kl_weight'],
        sparsity_weight=config['training']['sparsity_weight'],
        tf_weight=config['training']['tf_weight']
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        verbose=True
    )
    
    # Initialize trainer
    trainer = TAPESTRYTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device
    )
    
    # Train!
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['training']['epochs'],
        true_tf=true_tf,
        tf_mask=tf_mask,
        cancer_components=None,  # Don't know yet which are cancer
        save_dir=args.output_dir,
        early_stopping_patience=20
    )
    
    # Save final model
    logger.info("\nSaving final model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': trainer.history,
        'regions': regions.to_dict(),
        'n_samples': n_samples,
        'n_regions': methylation.shape[1],
    }, args.output_dir / 'final_model.pt')
    
    # Plot training curves
    logger.info("Plotting training curves...")
    trainer.plot_training_curves(args.output_dir / 'training_curves.png')
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()