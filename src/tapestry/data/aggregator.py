"""
Aggregate CpG-level data to genomic regions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)


class RegionAggregator:
    """
    Aggregate CpG methylation data to genomic regions.
    """
    
    def __init__(
        self,
        min_cpgs_per_region: int = 3,
        min_region_coverage: int = 20
    ):
        """
        Initialize aggregator.
        
        Args:
            min_cpgs_per_region: Minimum number of CpGs in a region
            min_region_coverage: Minimum total coverage for a region
        """
        self.min_cpgs_per_region = min_cpgs_per_region
        self.min_region_coverage = min_region_coverage
    
    def aggregate_sample(
        self,
        cpg_data: pd.DataFrame,
        regions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate CpG data for one sample to regions.
        
        Args:
            cpg_data: DataFrame with CpG-level data (chrom, pos, mod, coverage)
            regions: DataFrame with region definitions (chrom, start, end, region_id)
            
        Returns:
            DataFrame with region-level aggregation
        """
        # Sort both dataframes for efficient merging
        cpg_data = cpg_data.sort_values(['chrom', 'pos']).reset_index(drop=True)
        regions = regions.sort_values(['chrom', 'start']).reset_index(drop=True)
        
        results = []
        
        # Process chromosome by chromosome for memory efficiency
        for chrom in regions['chrom'].unique():
            chrom_cpgs = cpg_data[cpg_data['chrom'] == chrom]
            chrom_regions = regions[regions['chrom'] == chrom]
            
            if len(chrom_cpgs) == 0:
                continue
            
            # For each region, find overlapping CpGs
            for _, region in chrom_regions.iterrows():
                # Find CpGs within this region
                mask = (
                    (chrom_cpgs['pos'] >= region['start']) &
                    (chrom_cpgs['pos'] < region['end'])
                )
                
                region_cpgs = chrom_cpgs[mask]
                
                if len(region_cpgs) < self.min_cpgs_per_region:
                    continue
                
                # Aggregate counts
                total_mod = region_cpgs['mod'].sum()
                total_coverage = region_cpgs['coverage'].sum()
                
                if total_coverage < self.min_region_coverage:
                    continue
                
                # Calculate regional methylation rate
                region_rate = total_mod / total_coverage if total_coverage > 0 else 0
                
                results.append({
                    'region_id': region['region_id'],
                    'chrom': region['chrom'],
                    'start': region['start'],
                    'end': region['end'],
                    'n_cpgs': len(region_cpgs),
                    'mod_count': int(total_mod),
                    'coverage': int(total_coverage),
                    'methylation_rate': region_rate
                })
        
        return pd.DataFrame(results)
    
    def aggregate_cohort(
        self,
        cohort_data: Dict[str, pd.DataFrame],
        regions: pd.DataFrame,
        min_samples_covered: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """
        Aggregate entire cohort to regions.
        
        Args:
            cohort_data: Dictionary mapping sample_id -> CpG DataFrame
            regions: DataFrame with region definitions
            min_samples_covered: Minimum number of samples covering a region
            
        Returns:
            Tuple of:
                - methylation_matrix: [n_samples × n_regions] mod counts
                - coverage_matrix: [n_samples × n_regions] total coverage
                - regions_kept: DataFrame of regions that passed filters
                - sample_ids: List of sample IDs (row order)
        """
        logger.info(f"Aggregating {len(cohort_data)} samples to regions...")
        
        sample_ids = list(cohort_data.keys())
        n_samples = len(sample_ids)
        n_regions = len(regions)
        
        # Initialize matrices (sparse initially)
        region_data = defaultdict(lambda: {
            'mod': np.zeros(n_samples),
            'coverage': np.zeros(n_samples),
            'n_samples': 0
        })
        
        # Aggregate each sample
        for sample_idx, sample_id in enumerate(tqdm(sample_ids, desc="Aggregating")):
            cpg_data = cohort_data[sample_id]
            sample_regions = self.aggregate_sample(cpg_data, regions)
            
            # Store in dictionary
            for _, row in sample_regions.iterrows():
                region_id = row['region_id']
                region_data[region_id]['mod'][sample_idx] = row['mod_count']
                region_data[region_id]['coverage'][sample_idx] = row['coverage']
                region_data[region_id]['n_samples'] += 1
        
        # Filter regions by number of samples covered
        logger.info(f"Filtering regions (min {min_samples_covered} samples)...")
        
        kept_regions = []
        meth_arrays = []
        cov_arrays = []
        
        for region_id in region_data.keys():
            if region_data[region_id]['n_samples'] >= min_samples_covered:
                kept_regions.append(region_id)
                meth_arrays.append(region_data[region_id]['mod'])
                cov_arrays.append(region_data[region_id]['coverage'])
        
        # Convert to matrices
        methylation_matrix = np.array(meth_arrays).T  # [samples × regions]
        coverage_matrix = np.array(cov_arrays).T      # [samples × regions]
        
        # Filter regions dataframe
        regions_kept = regions[regions['region_id'].isin(kept_regions)].copy()
        regions_kept = regions_kept.set_index('region_id').loc[kept_regions].reset_index()
        
        logger.info(
            f"Kept {len(kept_regions)} regions "
            f"(coverage in ≥{min_samples_covered} samples)"
        )
        
        return methylation_matrix, coverage_matrix, regions_kept, sample_ids
    
    def filter_by_variance(
        self,
        methylation_matrix: np.ndarray,
        coverage_matrix: np.ndarray,
        regions: pd.DataFrame,
        top_k: int = 5000,
        min_variance: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Select most variable regions.
        
        Args:
            methylation_matrix: [n_samples × n_regions]
            coverage_matrix: [n_samples × n_regions]
            regions: DataFrame with region info
            top_k: Number of regions to keep
            min_variance: Minimum variance threshold
            
        Returns:
            Filtered matrices and regions
        """
        # Calculate beta values (methylation rates)
        beta = methylation_matrix / (coverage_matrix + 1e-6)
        
        # Calculate variance across samples
        variances = np.var(beta, axis=0)
        
        # Filter by minimum variance
        mask = variances >= min_variance
        
        logger.info(
            f"Regions with variance ≥ {min_variance}: "
            f"{mask.sum()} / {len(mask)}"
        )
        
        # Select top K by variance
        if mask.sum() > top_k:
            # Get indices of top K
            valid_indices = np.where(mask)[0]
            valid_variances = variances[valid_indices]
            top_indices = valid_indices[np.argsort(valid_variances)[-top_k:]]
        else:
            top_indices = np.where(mask)[0]
        
        logger.info(f"Selected {len(top_indices)} most variable regions")
        
        # Filter matrices and regions
        meth_filtered = methylation_matrix[:, top_indices]
        cov_filtered = coverage_matrix[:, top_indices]
        regions_filtered = regions.iloc[top_indices].reset_index(drop=True)
        
        # Log statistics
        mean_var = variances[top_indices].mean()
        logger.info(f"Mean variance of selected regions: {mean_var:.4f}")
        
        return meth_filtered, cov_filtered, regions_filtered