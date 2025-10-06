"""
Component interpretation and analysis for TAPESTRY.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging
from sklearn.metrics import roc_auc_score
from scipy import stats

logger = logging.getLogger(__name__)


class ComponentAnalyser:
    """
    Analyse and interpret learned components.
    """
    
    def __init__(
        self,
        signatures: np.ndarray,
        proportions: np.ndarray,
        regions: pd.DataFrame,
        sample_ids: List[str],
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Initialise analyser.
        
        Args:
            signatures: Learned signature matrix [n_components × n_regions]
            proportions: Component proportions [n_samples × n_components]
            regions: DataFrame with region annotations
            sample_ids: List of sample IDs
            metadata: Optional metadata with clinical variables
        """
        self.signatures = signatures
        self.proportions = proportions
        self.regions = regions
        self.sample_ids = sample_ids
        self.metadata = metadata
        
        self.n_components = signatures.shape[0]
        self.n_regions = signatures.shape[1]
        self.n_samples = proportions.shape[0]
        
        logger.info(
            f"Initialised analyser: {self.n_components} components, "
            f"{self.n_samples} samples, {self.n_regions} regions"
        )
    
    def compute_component_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for each component.
        
        Returns:
            DataFrame with statistics per component
        """
        stats_list = []
        
        for comp_idx in range(self.n_components):
            sig = self.signatures[comp_idx]
            props = self.proportions[:, comp_idx]
            
            # Methylation statistics
            mean_meth = sig.mean()
            std_meth = sig.std()
            median_meth = np.median(sig)
            
            # Bimodality
            low_meth = (sig < 0.2).mean()
            high_meth = (sig > 0.8).mean()
            bimodality = low_meth + high_meth
            
            # Proportion statistics
            mean_prop = props.mean()
            std_prop = props.std()
            max_prop = props.max()
            
            # Sparsity (how many samples have near-zero proportion)
            sparsity = (props < 0.01).mean()
            
            stats_list.append({
                'component': comp_idx,
                'mean_methylation': mean_meth,
                'std_methylation': std_meth,
                'median_methylation': median_meth,
                'bimodality_score': bimodality,
                'mean_proportion': mean_prop,
                'std_proportion': std_prop,
                'max_proportion': max_prop,
                'sparsity': sparsity,
            })
        
        return pd.DataFrame(stats_list)
    
    def identify_cancer_components(
        self,
        diagnosis_col: str = 'diagnosis',
        tf_col: str = 'ichorCNA_tf',
        auc_threshold: float = 0.75,
        correlation_threshold: float = 0.5
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Identify which components are cancer-related.
        
        Uses multiple criteria:
        1. High AUC for separating cancer vs healthy
        2. Strong correlation with tumour fraction
        3. Higher proportion in cancer samples
        
        Args:
            diagnosis_col: Column name for cancer/healthy labels
            tf_col: Column name for tumour fraction
            auc_threshold: Minimum AUC to consider cancer-related
            correlation_threshold: Minimum correlation with TF
            
        Returns:
            (cancer_component_indices, analysis_dataframe)
        """
        if self.metadata is None:
            logger.warning("No metadata provided, cannot identify cancer components")
            return [], pd.DataFrame()
        
        results = []
        
        for comp_idx in range(self.n_components):
            props = self.proportions[:, comp_idx]
            
            result = {'component': comp_idx}
            
            # 1. AUC for cancer classification
            if diagnosis_col in self.metadata.columns:
                diagnosis = self.metadata[diagnosis_col].values
                
                # Handle different encodings
                if diagnosis.dtype == object or diagnosis.dtype.name == 'category':
                    # Assume 'cancer' or similar vs 'healthy' or 'control'
                    cancer_labels = np.array([
                        1 if 'cancer' in str(d).lower() or 'tumour' in str(d).lower() 
                        else 0 
                        for d in diagnosis
                    ])
                else:
                    cancer_labels = diagnosis.astype(int)
                
                if len(np.unique(cancer_labels)) == 2:
                    auc = roc_auc_score(cancer_labels, props)
                    result['auc'] = auc
                    result['auc_significant'] = auc > auc_threshold
                else:
                    result['auc'] = np.nan
                    result['auc_significant'] = False
            else:
                result['auc'] = np.nan
                result['auc_significant'] = False
            
            # 2. Correlation with tumour fraction
            if tf_col in self.metadata.columns:
                tf = self.metadata[tf_col].values
                
                # Only use reliable TF estimates (> 5%)
                reliable_mask = tf > 0.05
                
                if reliable_mask.sum() > 5:
                    corr, p_value = stats.pearsonr(
                        props[reliable_mask],
                        tf[reliable_mask]
                    )
                    result['tf_correlation'] = corr
                    result['tf_pvalue'] = p_value
                    result['tf_significant'] = (
                        corr > correlation_threshold and p_value < 0.05
                    )
                else:
                    result['tf_correlation'] = np.nan
                    result['tf_pvalue'] = np.nan
                    result['tf_significant'] = False
            else:
                result['tf_correlation'] = np.nan
                result['tf_pvalue'] = np.nan
                result['tf_significant'] = False
            
            # 3. Mean proportion in cancer vs healthy
            if diagnosis_col in self.metadata.columns:
                cancer_mask = cancer_labels == 1
                healthy_mask = cancer_labels == 0
                
                if cancer_mask.sum() > 0 and healthy_mask.sum() > 0:
                    cancer_mean = props[cancer_mask].mean()
                    healthy_mean = props[healthy_mask].mean()
                    fold_change = cancer_mean / (healthy_mean + 1e-6)
                    
                    # T-test
                    t_stat, p_val = stats.ttest_ind(
                        props[cancer_mask],
                        props[healthy_mask]
                    )
                    
                    result['cancer_mean'] = cancer_mean
                    result['healthy_mean'] = healthy_mean
                    result['fold_change'] = fold_change
                    result['ttest_pvalue'] = p_val
                    result['enriched_in_cancer'] = (
                        cancer_mean > healthy_mean and p_val < 0.05
                    )
                else:
                    result['cancer_mean'] = np.nan
                    result['healthy_mean'] = np.nan
                    result['fold_change'] = np.nan
                    result['ttest_pvalue'] = np.nan
                    result['enriched_in_cancer'] = False
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Identify cancer components
        cancer_components = []
        
        for _, row in results_df.iterrows():
            is_cancer = (
                row.get('auc_significant', False) or
                row.get('tf_significant', False) or
                row.get('enriched_in_cancer', False)
            )
            
            if is_cancer:
                cancer_components.append(int(row['component']))
        
        logger.info(f"Identified {len(cancer_components)} cancer-related components: {cancer_components}")
        
        return cancer_components, results_df
    
    def get_top_regions_for_component(
        self,
        component_idx: int,
        n_top: int = 50,
        methylation_type: str = 'both'
    ) -> pd.DataFrame:
        """
        Get top hypomethylated and/or hypermethylated regions for a component.
        
        Args:
            component_idx: Component index
            n_top: Number of top regions to return
            methylation_type: 'hypo', 'hyper', or 'both'
            
        Returns:
            DataFrame with top regions
        """
        sig = self.signatures[component_idx]
        
        results = []
        
        if methylation_type in ['hypo', 'both']:
            # Top hypomethylated
            hypo_indices = np.argsort(sig)[:n_top]
            
            for idx in hypo_indices:
                region = self.regions.iloc[idx]
                results.append({
                    'component': component_idx,
                    'region_id': region['region_id'],
                    'chrom': region['chrom'],
                    'start': region['start'],
                    'end': region['end'],
                    'methylation': sig[idx],
                    'type': 'hypomethylated',
                    'rank': len([r for r in results if r['type'] == 'hypomethylated']) + 1
                })
        
        if methylation_type in ['hyper', 'both']:
            # Top hypermethylated
            hyper_indices = np.argsort(sig)[-n_top:][::-1]
            
            for idx in hyper_indices:
                region = self.regions.iloc[idx]
                results.append({
                    'component': component_idx,
                    'region_id': region['region_id'],
                    'chrom': region['chrom'],
                    'start': region['start'],
                    'end': region['end'],
                    'methylation': sig[idx],
                    'type': 'hypermethylated',
                    'rank': len([r for r in results if r['type'] == 'hypermethylated']) + 1
                })
        
        return pd.DataFrame(results)
    
    def compute_sample_cancer_scores(
        self,
        cancer_components: List[int]
    ) -> np.ndarray:
        """
        Compute cancer score for each sample as sum of cancer component proportions.
        
        Args:
            cancer_components: List of cancer component indices
            
        Returns:
            Array of cancer scores [n_samples]
        """
        if len(cancer_components) == 0:
            logger.warning("No cancer components specified")
            return np.zeros(self.n_samples)
        
        cancer_scores = self.proportions[:, cancer_components].sum(axis=1)
        
        logger.info(
            f"Computed cancer scores (mean={cancer_scores.mean():.4f}, "
            f"std={cancer_scores.std():.4f})"
        )
        
        return cancer_scores
    
    def evaluate_performance(
        self,
        cancer_scores: np.ndarray,
        diagnosis_col: str = 'diagnosis',
        tf_col: str = 'ichorCNA_tf'
    ) -> Dict:
        """
        Evaluate performance of cancer detection.
        
        Args:
            cancer_scores: Predicted cancer scores
            diagnosis_col: Column for cancer/healthy labels
            tf_col: Column for tumour fraction
            
        Returns:
            Dictionary with performance metrics
        """
        if self.metadata is None:
            logger.warning("No metadata for evaluation")
            return {}
        
        metrics = {}
        
        # Classification performance
        if diagnosis_col in self.metadata.columns:
            diagnosis = self.metadata[diagnosis_col].values
            
            if diagnosis.dtype == object or diagnosis.dtype.name == 'category':
                cancer_labels = np.array([
                    1 if 'cancer' in str(d).lower() or 'tumour' in str(d).lower() 
                    else 0 
                    for d in diagnosis
                ])
            else:
                cancer_labels = diagnosis.astype(int)
            
            if len(np.unique(cancer_labels)) == 2:
                from sklearn.metrics import roc_auc_score, roc_curve
                
                auc = roc_auc_score(cancer_labels, cancer_scores)
                fpr, tpr, thresholds = roc_curve(cancer_labels, cancer_scores)
                
                # Find optimal threshold (Youden's index)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                metrics['auc'] = auc
                metrics['optimal_threshold'] = optimal_threshold
                metrics['sensitivity_at_optimal'] = tpr[optimal_idx]
                metrics['specificity_at_optimal'] = 1 - fpr[optimal_idx]
                
                logger.info(f"Classification AUC: {auc:.3f}")
        
        # Regression performance (TF correlation)
        if tf_col in self.metadata.columns:
            tf = self.metadata[tf_col].values
            
            # Overall correlation
            valid_mask = ~np.isnan(tf)
            if valid_mask.sum() > 5:
                corr_all, p_all = stats.pearsonr(
                    cancer_scores[valid_mask],
                    tf[valid_mask]
                )
                metrics['tf_correlation_all'] = corr_all
                metrics['tf_pvalue_all'] = p_all
                
                logger.info(f"TF correlation (all samples): {corr_all:.3f}")
            
            # High-TF samples only
            reliable_mask = tf > 0.05
            if reliable_mask.sum() > 5:
                corr_high, p_high = stats.pearsonr(
                    cancer_scores[reliable_mask],
                    tf[reliable_mask]
                )
                
                mae = np.mean(np.abs(
                    cancer_scores[reliable_mask] - tf[reliable_mask]
                ))
                
                metrics['tf_correlation_high'] = corr_high
                metrics['tf_pvalue_high'] = p_high
                metrics['mae_high_tf'] = mae
                
                logger.info(f"TF correlation (TF > 5%): {corr_high:.3f}, MAE: {mae:.4f}")
        
        return metrics