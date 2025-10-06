"""
Load TAPS methylation data from sample files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TAPSLoader:
    """
    Load and parse TAPS methylation data files.
    
    Expected format:
    CHROM  START  END  STRAND  RATE  UNMOD  MOD  CLASS  CONTEXT
    """
    
    def __init__(
        self,
        min_coverage: int = 5,
        context_filter: List[str] = ['CpG']
    ):
        """
        Initialize TAPS data loader.
        
        Args:
            min_coverage: Minimum coverage threshold for a site
            context_filter: List of contexts to keep (e.g., ['CpG'])
        """
        self.min_coverage = min_coverage
        self.context_filter = context_filter
        
    def load_sample(
        self, 
        filepath: Path
    ) -> pd.DataFrame:
        """
        Load a single TAPS sample file.
        
        Args:
            filepath: Path to sample file
            
        Returns:
            DataFrame with columns: chrom, pos, mod, coverage, rate
        """
        try:
            # Read file
            df = pd.read_csv(
                filepath,
                sep='\t',
                comment='#',
                names=['chrom', 'start', 'end', 'strand', 'rate', 
                       'unmod', 'mod', 'class', 'context']
            )
            
            # Filter to desired context (e.g., CpG only)
            if self.context_filter:
                df = df[df['context'].isin(self.context_filter)].copy()
            
            # Calculate coverage
            df['coverage'] = df['unmod'] + df['mod']
            
            # Filter by minimum coverage
            df = df[df['coverage'] >= self.min_coverage].copy()
            
            # Use start position as CpG position
            df['pos'] = df['start']
            
            # Keep only necessary columns
            df = df[['chrom', 'pos', 'mod', 'coverage']].copy()
            
            # Calculate rate (handle division by zero)
            df['rate'] = df['mod'] / df['coverage']
            
            logger.debug(
                f"Loaded {len(df)} CpG sites from {filepath.name}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def load_cohort(
        self,
        sample_files: List[Path],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple TAPS sample files.
        
        Args:
            sample_files: List of paths to sample files
            sample_ids: Optional list of sample IDs (defaults to filenames)
            
        Returns:
            Dictionary mapping sample_id -> DataFrame
        """
        if sample_ids is None:
            sample_ids = [f.stem for f in sample_files]
        
        if len(sample_ids) != len(sample_files):
            raise ValueError(
                f"Number of sample IDs ({len(sample_ids)}) must match "
                f"number of files ({len(sample_files)})"
            )
        
        cohort_data = {}
        
        logger.info(f"Loading {len(sample_files)} samples...")
        
        for sample_id, filepath in tqdm(
            zip(sample_ids, sample_files),
            total=len(sample_files),
            desc="Loading samples"
        ):
            df = self.load_sample(filepath)
            cohort_data[sample_id] = df
            
        # Log summary statistics
        total_sites = sum(len(df) for df in cohort_data.values())
        avg_sites = total_sites / len(cohort_data)
        
        logger.info(
            f"Loaded {len(cohort_data)} samples with "
            f"{avg_sites:.0f} CpG sites per sample (avg)"
        )
        
        return cohort_data
    
    def get_coverage_stats(
        self,
        cohort_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate coverage statistics across cohort.
        
        Args:
            cohort_data: Dictionary of sample DataFrames
            
        Returns:
            DataFrame with coverage statistics per sample
        """
        stats = []
        
        for sample_id, df in cohort_data.items():
            stats.append({
                'sample_id': sample_id,
                'n_sites': len(df),
                'total_coverage': df['coverage'].sum(),
                'mean_coverage': df['coverage'].mean(),
                'median_coverage': df['coverage'].median(),
                'mean_methylation': df['rate'].mean(),
            })
        
        return pd.DataFrame(stats)