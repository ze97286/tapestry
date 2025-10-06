"""
Component interpretation and analysis for TAPESTRY.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """
    Analyze and interpret learned components from blind deconvolution.
    """
    
    def __init__(
        self,
        model,
        methylation: np.ndarray,
        coverage: np.ndarray,
        regions: pd.DataFrame,
        sample_ids: List[str],
        metadata: pd.DataFrame = None,
        device: str = 'cuda'
    ):
        """
        Initialize analyzer.
        
        Args:
            model: Trained BlindDeconvolutionVAE
            methylation: Methylation counts [n_samples × n_regions]
            coverage: Coverage [n_samples × n_regions]
            regions: Region annotations
            sample_ids: List of sample IDs
            metadata: Sample metadata (diagnosis, TF, etc.)
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.methylation = methylation
        self.coverage = coverage
        self.regions = regions
        self.sample_ids = sample_ids
        self.metadata = metadata
        
        self.n_components = model.n_components
        self.n_regions = model.n_regions
        
        # Get proportions for all samples
        logger.info("Computing component proportions for all samples...")
        self.proportions = self._compute_all_proportions()
        
        # Get learned signatures
        self.signatures = model.signatures.cpu().numpy()
        
        logger.info(f"Initialized analyzer for {len(sample_ids)} samples")
    
    def _compute_all_proportions(self) -> np.ndarray:
        """
        Compute component proportions for all samples.
        
        Returns:
            Proportions [n_samples × n_components]
        """
        meth_tensor = torch.FloatTensor(self.methylation).to(self.device)
        cov_tensor = torch.FloatTensor(self.coverage).to(self.device)
        
        with torch.no_grad():
            proportions = self.model.get_proportions(meth_tensor, cov_tensor)
        
        return proportions.cpu().numpy()
    
    def identify_cancer_components(
        self,
        min_auc: float = 0.70,
        min_correlation: float = 0.50
    ) -> Dict[int, Dict]:
        """
        Identify which components are cancer-related.
        
        Uses multiple criteria:
        1. AUC for separating cancer vs healthy (if diagnosis available)
        2. Correlation with tumor fraction (if TF available)
        3. Variance across samples (cancer should vary)
        
        Args:
            min_auc: Minimum AUC to consider cancer-related
            min_correlation: Minimum TF correlation to consider cancer-related
            
        Returns:
            Dictionary mapping component_idx -> interpretation dict
        """
        logger.info("\nIdentifying cancer-related components...")
        
        if self.metadata is None:
            logger.warning("No metadata - cannot identify cancer components")
            return {}
        
        results = {}
        
        for comp_idx in range(self.n_components):
            comp_props = self.proportions[:, comp_idx]
            
            interpretation = {
                'component_idx': comp_idx,
                'mean_proportion': comp_props.mean(),
                'std_proportion': comp_props.std(),
                'is_cancer': False,
                'evidence': []
            }
            
            # Criterion 1: AUC for cancer vs healthy
            if 'diagnosis' in self.metadata.columns:
                diagnosis = self.metadata['diagnosis'].values
                
                # Handle different label formats
                if diagnosis.dtype == 'object':
                    # Convert to binary (1 = cancer, 0 = healthy)
                    cancer_mask = diagnosis.isin(['cancer', 'Cancer', 1, '1'])
                    diagnosis_binary = cancer_mask.astype(int)
                else:
                    diagnosis_binary = diagnosis
                
                if len(np.unique(diagnosis_binary)) > 1:  # Need both classes
                    auc = roc_auc_score(diagnosis_binary, comp_props)
                    interpretation['auc'] = auc
                    
                    if auc > min_auc:
                        interpretation['is_cancer'] = True
                        interpretation['evidence'].append(f'AUC={auc:.3f}')
                        logger.info(f"  Component {comp_idx}: AUC={auc:.3f} ✓ Cancer-related")
                    elif auc < (1 - min_auc):
                        # Inverse correlation (low in cancer)
                        interpretation['evidence'].append(f'AUC={auc:.3f} (inverse)')
                        logger.info(f"  Component {comp_idx}: AUC={auc:.3f} (normal cfDNA)")
                    else:
                        logger.debug(f"  Component {comp_idx}: AUC={auc:.3f} (neutral)")
            
            # Criterion 2: Correlation with tumor fraction
            if 'ichorCNA_tf' in self.metadata.columns:
                tf = self.metadata['ichorCNA_tf'].values
                
                # Only use reliable estimates (TF > 5%)
                reliable_mask = tf > 0.05
                
                if reliable_mask.sum() > 5:  # Need enough samples
                    corr, pval = pearsonr(
                        comp_props[reliable_mask],
                        tf[reliable_mask]
                    )
                    interpretation['tf_correlation'] = corr
                    interpretation['tf_pvalue'] = pval
                    
                    if abs(corr) > min_correlation and pval < 0.05:
                        if corr > 0:  # Positive correlation
                            interpretation['is_cancer'] = True
                            interpretation['evidence'].append(f'TF_corr={corr:.3f}')
                            logger.info(f"  Component {comp_idx}: TF correlation={corr:.3f} ✓ Cancer-related")
                        else:  # Negative correlation
                            interpretation['evidence'].append(f'TF_corr={corr:.3f} (depleted)')
            
            # Criterion 3: Variance (cancer should vary across patients)
            interpretation['variance_rank'] = np.argsort(
                -self.proportions.std(axis=0)
            ).tolist().index(comp_idx)
            
            results[comp_idx] = interpretation
        
        # Identify cancer components
        cancer_components = [
            idx for idx, info in results.items()
            if info['is_cancer']
        ]
        
        logger.info(f"\nIdentified {len(cancer_components)} cancer-related components: {cancer_components}")
        
        return results
    
    def analyze_signatures(
        self,
        component_idx: int,
        top_k: int = 20
    ) -> Dict:
        """
        Analyze methylation signature for a component.
        
        Args:
            component_idx: Component to analyze
            top_k: Number of top regions to report
            
        Returns:
            Dictionary with signature analysis
        """
        sig = self.signatures[component_idx]
        
        # Find highly/lowly methylated regions
        high_indices = np.argsort(sig)[-top_k:][::-1]
        low_indices = np.argsort(sig)[:top_k]
        
        high_regions = self.regions.iloc[high_indices].copy()
        high_regions['methylation'] = sig[high_indices]
        
        low_regions = self.regions.iloc[low_indices].copy()
        low_regions['methylation'] = sig[low_indices]
        
        analysis = {
            'component_idx': component_idx,
            'mean_methylation': sig.mean(),
            'median_methylation': np.median(sig),
            'high_methylated_regions': high_regions,
            'low_methylated_regions': low_regions,
            'bimodality_score': self._compute_bimodality(sig)
        }
        
        return analysis
    
    def _compute_bimodality(self, values: np.ndarray) -> float:
        """
        Compute bimodality score (how much signal is at extremes).
        
        Args:
            values: Array of methylation values
            
        Returns:
            Score between 0 (uniform) and 1 (bimodal)
        """
        low = (values < 0.2).mean()
        high = (values > 0.8).mean()
        return low + high
    
    def validate_performance(
        self,
        test_indices: np.ndarray,
        cancer_components: List[int]
    ) -> Dict:
        """
        Validate model performance on held-out test set.
        
        Args:
            test_indices: Indices of test samples
            cancer_components: List of cancer component indices
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("\nValidating model performance on test set...")
        
        if self.metadata is None:
            logger.warning("No metadata - cannot validate")
            return {}
        
        # Get test data
        test_proportions = self.proportions[test_indices]
        test_metadata = self.metadata.iloc[test_indices]
        
        # Cancer score = sum of cancer components
        cancer_scores = test_proportions[:, cancer_components].sum(axis=1)
        
        metrics = {}
        
        # Binary classification (if diagnosis available)
        if 'diagnosis' in test_metadata.columns:
            diagnosis = test_metadata['diagnosis'].values
            
            if diagnosis.dtype == 'object':
                cancer_mask = diagnosis.isin(['cancer', 'Cancer', 1, '1'])
                diagnosis_binary = cancer_mask.astype(int)
            else:
                diagnosis_binary = diagnosis
            
            if len(np.unique(diagnosis_binary)) > 1:
                # AUC
                auc = roc_auc_score(diagnosis_binary, cancer_scores)
                metrics['auc'] = auc
                
                # ROC curve
                fpr, tpr, thresholds = roc_curve(diagnosis_binary, cancer_scores)
                
                # Optimal threshold (Youden's index)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                metrics['optimal_threshold'] = optimal_threshold
                metrics['sensitivity'] = tpr[optimal_idx]
                metrics['specificity'] = 1 - fpr[optimal_idx]
                
                logger.info(f"  Test AUC: {auc:.3f}")
                logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
                logger.info(f"  Sensitivity: {tpr[optimal_idx]:.3f}")
                logger.info(f"  Specificity: {1-fpr[optimal_idx]:.3f}")
        
        # Quantitative evaluation (if TF available)
        if 'ichorCNA_tf' in test_metadata.columns:
            tf_true = test_metadata['ichorCNA_tf'].values
            
            # Overall correlation
            corr_all, _ = pearsonr(cancer_scores, tf_true)
            metrics['tf_correlation_all'] = corr_all
            
            # Reliable samples only (TF > 5%)
            reliable_mask = tf_true > 0.05
            
            if reliable_mask.sum() > 5:
                tf_reliable = tf_true[reliable_mask]
                pred_reliable = cancer_scores[reliable_mask]
                
                corr_reliable, _ = pearsonr(pred_reliable, tf_reliable)
                mae = mean_absolute_error(tf_reliable, pred_reliable)
                r2 = r2_score(tf_reliable, pred_reliable)
                
                metrics['tf_correlation_reliable'] = corr_reliable
                metrics['mae'] = mae
                metrics['r2'] = r2
                
                logger.info(f"  TF correlation (all): {corr_all:.3f}")
                logger.info(f"  TF correlation (TF>5%): {corr_reliable:.3f}")
                logger.info(f"  MAE (TF>5%): {mae:.4f}")
                logger.info(f"  R² (TF>5%): {r2:.3f}")
                
                # Stratified MAE
                low_tf = (tf_true > 0.001) & (tf_true < 0.01)
                med_tf = (tf_true >= 0.01) & (tf_true < 0.05)
                high_tf = tf_true >= 0.05
                
                if low_tf.sum() > 0:
                    mae_low = mean_absolute_error(
                        tf_true[low_tf],
                        cancer_scores[low_tf]
                    )
                    metrics['mae_low_tf'] = mae_low
                    logger.info(f"  MAE (0.1-1% TF): {mae_low:.4f}")
                
                if med_tf.sum() > 0:
                    mae_med = mean_absolute_error(
                        tf_true[med_tf],
                        cancer_scores[med_tf]
                    )
                    metrics['mae_med_tf'] = mae_med
                    logger.info(f"  MAE (1-5% TF): {mae_med:.4f}")
                
                if high_tf.sum() > 0:
                    mae_high = mean_absolute_error(
                        tf_true[high_tf],
                        cancer_scores[high_tf]
                    )
                    metrics['mae_high_tf'] = mae_high
                    logger.info(f"  MAE (>5% TF): {mae_high:.4f}")
        
        # Healthy control check
        if 'diagnosis' in test_metadata.columns:
            healthy_mask = diagnosis_binary == 0
            
            if healthy_mask.sum() > 0:
                healthy_scores = cancer_scores[healthy_mask]
                metrics['healthy_mean_score'] = healthy_scores.mean()
                metrics['healthy_max_score'] = healthy_scores.max()
                
                logger.info(f"  Healthy controls:")
                logger.info(f"    Mean cancer score: {healthy_scores.mean():.4f}")
                logger.info(f"    Max cancer score: {healthy_scores.max():.4f}")
                logger.info(f"    All < 0.001: {(healthy_scores < 0.001).all()}")
        
        return metrics
    
    def plot_component_summary(
        self,
        component_interpretations: Dict,
        save_path: str = None
    ):
        """
        Create summary plot of all components.
        
        Args:
            component_interpretations: From identify_cancer_components()
            save_path: Path to save figure
        """
        n_comp = self.n_components
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Mean proportions across components
        ax = axes[0, 0]
        means = [component_interpretations[i]['mean_proportion'] for i in range(n_comp)]
        stds = [component_interpretations[i]['std_proportion'] for i in range(n_comp)]
        colors = ['red' if component_interpretations[i]['is_cancer'] else 'steelblue' 
                  for i in range(n_comp)]
        
        ax.bar(range(n_comp), means, yerr=stds, color=colors, alpha=0.7)
        ax.set_xlabel('Component')
        ax.set_ylabel('Mean Proportion')
        ax.set_title('Component Proportions Across Samples')
        ax.set_xticks(range(n_comp))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Cancer-related'),
            Patch(facecolor='steelblue', alpha=0.7, label='Normal/Other')
        ]
        ax.legend(handles=legend_elements)
        
        # Plot 2: AUC values (if available)
        ax = axes[0, 1]
        aucs = [component_interpretations[i].get('auc', 0.5) for i in range(n_comp)]
        colors = ['red' if auc > 0.7 else 'steelblue' for auc in aucs]
        
        ax.barh(range(n_comp), aucs, color=colors, alpha=0.7)
        ax.axvline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.axvline(0.5, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Component')
        ax.set_xlabel('AUC (Cancer vs Healthy)')
        ax.set_title('Classification Performance')
        ax.set_xlim(0, 1)
        ax.set_yticks(range(n_comp))
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        # Plot 3: TF correlation (if available)
        ax = axes[1, 0]
        corrs = [component_interpretations[i].get('tf_correlation', 0) for i in range(n_comp)]
        colors = ['red' if abs(corr) > 0.5 else 'steelblue' for corr in corrs]
        
        ax.barh(range(n_comp), corrs, color=colors, alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.axvline(-0.5, color='red', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Component')
        ax.set_xlabel('Correlation with Tumor Fraction')
        ax.set_title('TF Correlation (for TF > 5%)')
        ax.set_xlim(-1, 1)
        ax.set_yticks(range(n_comp))
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        # Plot 4: Proportion distribution heatmap
        ax = axes[1, 1]
        
        # Sort samples by total cancer score
        cancer_comps = [i for i, info in component_interpretations.items() if info['is_cancer']]
        if cancer_comps:
            cancer_scores = self.proportions[:, cancer_comps].sum(axis=1)
            sorted_indices = np.argsort(cancer_scores)
        else:
            sorted_indices = np.arange(len(self.sample_ids))
        
        # Plot heatmap (subsample if too many samples)
        n_samples_plot = min(50, len(self.sample_ids))
        step = len(self.sample_ids) // n_samples_plot
        plot_indices = sorted_indices[::step]
        
        im = ax.imshow(
            self.proportions[plot_indices].T,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )
        ax.set_xlabel('Samples (sorted by cancer score)')
        ax.set_ylabel('Component')
        ax.set_title('Proportion Heatmap')
        ax.set_yticks(range(n_comp))
        plt.colorbar(im, ax=ax, label='Proportion')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved component summary to {save_path}")
        
        plt.close()
    
    def plot_calibration(
        self,
        cancer_components: List[int],
        test_indices: np.ndarray = None,
        save_path: str = None
    ):
        """
        Plot calibration of predicted TF vs actual TF.
        
        Args:
            cancer_components: List of cancer component indices
            test_indices: Indices to plot (if None, use all)
            save_path: Path to save figure
        """
        if self.metadata is None or 'ichorCNA_tf' not in self.metadata.columns:
            logger.warning("Cannot plot calibration - no TF data")
            return
        
        # Get data
        if test_indices is not None:
            proportions = self.proportions[test_indices]
            metadata = self.metadata.iloc[test_indices]
            title_suffix = " (Test Set)"
        else:
            proportions = self.proportions
            metadata = self.metadata
            title_suffix = ""
        
        # Cancer score
        cancer_scores = proportions[:, cancer_components].sum(axis=1)
        tf_true = metadata['ichorCNA_tf'].values
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Scatter plot
        ax = axes[0]
        
        # Color by reliability
        reliable = tf_true > 0.05
        
        ax.scatter(
            tf_true[~reliable], 
            cancer_scores[~reliable],
            alpha=0.3, 
            s=30, 
            c='lightgray',
            label='Low TF (unreliable)'
        )
        ax.scatter(
            tf_true[reliable], 
            cancer_scores[reliable],
            alpha=0.6, 
            s=50, 
            c='steelblue',
            label='High TF (reliable)'
        )
        
        # Perfect calibration line
        max_val = max(tf_true.max(), cancer_scores.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect calibration')
        
        # Correlation for reliable samples
        if reliable.sum() > 5:
            corr, _ = pearsonr(cancer_scores[reliable], tf_true[reliable])
            ax.text(
                0.05, 0.95,
                f'r = {corr:.3f}\n(TF > 5%)',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel('True Tumor Fraction (ichorCNA)')
        ax.set_ylabel('Predicted Cancer Score (TAPESTRY)')
        ax.set_title(f'Calibration{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax = axes[1]
        
        residuals = cancer_scores - tf_true
        
        ax.scatter(tf_true[~reliable], residuals[~reliable], alpha=0.3, s=30, c='lightgray')
        ax.scatter(tf_true[reliable], residuals[reliable], alpha=0.6, s=50, c='steelblue')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # MAE for reliable samples
        if reliable.sum() > 0:
            mae = np.abs(residuals[reliable]).mean()
            ax.text(
                0.05, 0.95,
                f'MAE = {mae:.4f}\n(TF > 5%)',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel('True Tumor Fraction')
        ax.set_ylabel('Residual (Predicted - True)')
        ax.set_title(f'Prediction Residuals{title_suffix}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved calibration plot to {save_path}")
        
        plt.close()
    
    def export_results(
        self,
        component_interpretations: Dict,
        cancer_components: List[int],
        output_dir: Path
    ):
        """
        Export all analysis results to CSV files.
        
        Args:
            component_interpretations: Component analysis results
            cancer_components: List of cancer component indices
            output_dir: Directory to save results
        """
        logger.info("\nExporting results...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Component summary
        comp_summary = []
        for idx, info in component_interpretations.items():
            comp_summary.append({
                'component': idx,
                'is_cancer': info['is_cancer'],
                'mean_proportion': info['mean_proportion'],
                'std_proportion': info['std_proportion'],
                'auc': info.get('auc', np.nan),
                'tf_correlation': info.get('tf_correlation', np.nan),
                'evidence': '; '.join(info['evidence'])
            })
        
        pd.DataFrame(comp_summary).to_csv(
            output_dir / 'component_summary.csv',
            index=False
        )
        logger.info(f"  Saved component_summary.csv")
        
        # 2. Sample proportions
        props_df = pd.DataFrame(
            self.proportions,
            columns=[f'component_{i}' for i in range(self.n_components)]
        )
        props_df.insert(0, 'sample_id', self.sample_ids)
        
        # Add cancer score
        props_df['cancer_score'] = self.proportions[:, cancer_components].sum(axis=1)
        
        # Add metadata if available
        if self.metadata is not None:
            for col in self.metadata.columns:
                if col not in props_df.columns:
                    props_df[col] = self.metadata[col].values
        
        props_df.to_csv(output_dir / 'sample_proportions.csv', index=False)
        logger.info(f"  Saved sample_proportions.csv")
        
        # 3. Signatures
        sigs_df = pd.DataFrame(
            self.signatures.T,
            columns=[f'component_{i}' for i in range(self.n_components)]
        )
        
        # Add region annotations
        for col in self.regions.columns:
            sigs_df[col] = self.regions[col].values
        
        sigs_df.to_csv(output_dir / 'signatures.csv', index=False)
        logger.info(f"  Saved signatures.csv")
        
        # 4. Cancer component details
        for comp_idx in cancer_components:
            sig_analysis = self.analyze_signatures(comp_idx, top_k=50)
            
            # High methylation regions
            sig_analysis['high_methylated_regions'].to_csv(
                output_dir / f'component_{comp_idx}_high_methylation.csv',
                index=False
            )
            
            # Low methylation regions
            sig_analysis['low_methylated_regions'].to_csv(
                output_dir / f'component_{comp_idx}_low_methylation.csv',
                index=False
            )
        
        logger.info(f"  Saved cancer component signatures")
        
        logger.info(f"\nAll results exported to: {output_dir}")