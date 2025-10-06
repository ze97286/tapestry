"""
Training loop for TAPESTRY.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TAPESTRYTrainer:
    """
    Trainer for blind deconvolution VAE.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: BlindDeconvolutionVAE model
            loss_fn: TAPESTRYLoss instance
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        true_tf: np.ndarray = None,
        tf_mask: np.ndarray = None,
        cancer_components: list = None
    ) -> dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            true_tf: Tumor fractions (optional)
            tf_mask: Mask for reliable TF estimates (optional)
            cancer_components: Indices of cancer components (optional)
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_divergence': 0.0,
            'sparsity': 0.0,
            'signature_reg': 0.0,
            'tumor_fraction': 0.0,
        }
        
        n_batches = 0
        
        for batch_idx, (meth, cov, indices) in enumerate(train_loader):
            meth = meth.to(self.device)
            cov = cov.to(self.device)
            
            # Get TF for this batch (if available)
            batch_tf = None
            batch_tf_mask = None
            if true_tf is not None:
                batch_tf = torch.FloatTensor(true_tf[indices]).to(self.device)
                batch_tf_mask = torch.BoolTensor(tf_mask[indices]).to(self.device)
            
            # Forward pass
            recon_beta, proportions, mu, logvar = self.model(meth, cov)
            
            # Compute loss
            losses = self.loss_fn(
                recon_beta, meth, cov,
                proportions, self.model.signatures,
                mu, logvar,
                true_tf=batch_tf,
                tf_mask=batch_tf_mask,
                cancer_components=cancer_components
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key] if key == 'total' else losses.get(key, 0.0)
            
            n_batches += 1
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def validate(
        self,
        val_loader: DataLoader,
        true_tf: np.ndarray = None,
        tf_mask: np.ndarray = None,
        cancer_components: list = None
    ) -> dict:
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            true_tf: Tumor fractions (optional)
            tf_mask: Mask for reliable TF estimates (optional)
            cancer_components: Indices of cancer components (optional)
            
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_divergence': 0.0,
            'sparsity': 0.0,
            'signature_reg': 0.0,
            'tumor_fraction': 0.0,
        }
        
        n_batches = 0
        
        with torch.no_grad():
            for batch_idx, (meth, cov, indices) in enumerate(val_loader):
                meth = meth.to(self.device)
                cov = cov.to(self.device)
                
                # Get TF for this batch (if available)
                batch_tf = None
                batch_tf_mask = None
                if true_tf is not None:
                    batch_tf = torch.FloatTensor(true_tf[indices]).to(self.device)
                    batch_tf_mask = torch.BoolTensor(tf_mask[indices]).to(self.device)
                
                # Forward pass
                recon_beta, proportions, mu, logvar = self.model(meth, cov)
                
                # Compute loss
                losses = self.loss_fn(
                    recon_beta, meth, cov,
                    proportions, self.model.signatures,
                    mu, logvar,
                    true_tf=batch_tf,
                    tf_mask=batch_tf_mask,
                    cancer_components=cancer_components
                )
                
                # Accumulate losses
                for key in epoch_losses.keys():
                    epoch_losses[key] += losses[key] if key == 'total' else losses.get(key, 0.0)
                
                n_batches += 1
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        true_tf: np.ndarray = None,
        tf_mask: np.ndarray = None,
        cancer_components: list = None,
        save_dir: Path = None,
        early_stopping_patience: int = 20
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            true_tf: Tumor fractions for all samples
            tf_mask: Mask for reliable TF estimates
            cancer_components: Indices of cancer components
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Training for {n_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_losses = self.train_epoch(
                train_loader, true_tf, tf_mask, cancer_components
            )
            
            # Validate
            val_losses = self.validate(
                val_loader, true_tf, tf_mask, cancer_components
            )
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_losses['total'])
            
            # Store history (detach if tensor)
            self.history['train_loss'].append(
                train_losses['total'].item() if isinstance(train_losses['total'], torch.Tensor) else train_losses['total']
            )
            self.history['val_loss'].append(
                val_losses['total'].item() if isinstance(val_losses['total'], torch.Tensor) else val_losses['total']
            )
            self.history['train_recon'].append(train_losses['reconstruction'])
            self.history['val_recon'].append(val_losses['reconstruction'])
            self.history['train_kl'].append(train_losses['kl_divergence'])
            self.history['val_kl'].append(val_losses['kl_divergence'])
            
            # Logging
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}/{n_epochs}: "
                    f"Train Loss={train_losses['total']:.1f}, "
                    f"Val Loss={val_losses['total']:.1f}, "
                    f"Recon={val_losses['reconstruction']:.1f}, "
                    f"KL={val_losses['kl_divergence']:.2f}"
                )
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                if save_dir is not None:
                    save_path = save_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'history': self.history,
                    }, save_path)
                    logger.info(f"Saved best model (val_loss={best_val_loss:.1f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training complete!")
    
    def plot_training_curves(self, save_path: Path = None):
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Total loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(self.history['train_recon'], label='Train')
        axes[1].plot(self.history['val_recon'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction (Binomial NLL)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.close()