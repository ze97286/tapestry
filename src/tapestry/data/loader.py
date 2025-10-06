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
            # Define dtypes explicitly to avoid type inference overhead
            dtype_spec = {
                'chrom': 'category',  # Much faster than string for chromosomes
                'start': 'int32',
                'end': 'int32',
                'strand': 'category',
                'rate': 'float32',
                'unmod': 'int32',
                'mod': 'int32',
                'class': 'category',
                'context': 'category'
            }

            # Only load columns we need (avoid loading/parsing unused data)
            usecols = ['chrom', 'start', 'unmod', 'mod', 'context']

            # Read file with optimizations
            df = pd.read_csv(
                filepath,
                sep='\t',
                comment='#',
                names=['chrom', 'start', 'end', 'strand', 'rate',
                       'unmod', 'mod', 'class', 'context'],
                dtype=dtype_spec,
                usecols=usecols,
                engine='c',  # Use C engine (faster)
                low_memory=False  # Avoid mixed type warnings
            )

            # Filter to desired context (e.g., CpG only) - do this BEFORE calculations
            if self.context_filter:
                df = df[df['context'].isin(self.context_filter)]

            # Calculate coverage in-place
            df['coverage'] = df['unmod'] + df['mod']

            # Filter by minimum coverage
            df = df[df['coverage'] >= self.min_coverage]

            # Rename start to pos (in-place)
            df.rename(columns={'start': 'pos'}, inplace=True)

            # Drop columns we don't need anymore
            df.drop(columns=['unmod', 'context'], inplace=True)

            # Calculate rate (already float32)
            df['rate'] = df['mod'] / df['coverage']

            # Convert chrom to string categories to save memory
            df['chrom'] = df['chrom'].astype('category')

            logger.debug(
                f"Loaded {len(df):,} CpG sites from {filepath.name}"
            )

            return df

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def iter_samples(
        self,
        sample_files: List[Path],
        sample_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ):
        """
        Generator that yields samples one at a time (memory efficient).

        Args:
            sample_files: List of paths to sample files
            sample_ids: Optional list of sample IDs (defaults to filenames)
            show_progress: Whether to show progress bar

        Yields:
            Tuple of (sample_id, DataFrame)
        """
        if sample_ids is None:
            sample_ids = [f.stem for f in sample_files]

        if len(sample_ids) != len(sample_files):
            raise ValueError(
                f"Number of sample IDs ({len(sample_ids)}) must match "
                f"number of files ({len(sample_files)})"
            )

        iterator = zip(sample_ids, sample_files)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(sample_files),
                desc="Processing samples"
            )

        for sample_id, filepath in iterator:
            df = self.load_sample(filepath)
            yield sample_id, df

    def load_cohort(
        self,
        sample_files: List[Path],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple TAPS sample files.

        WARNING: This loads ALL samples into memory at once.
        For large cohorts (>10 samples), use iter_samples() instead.

        Args:
            sample_files: List of paths to sample files
            sample_ids: Optional list of sample IDs (defaults to filenames)

        Returns:
            Dictionary mapping sample_id -> DataFrame
        """
        logger.warning(
            f"Loading {len(sample_files)} samples into memory. "
            "Consider using iter_samples() for large cohorts."
        )

        cohort_data = {}

        for sample_id, df in self.iter_samples(sample_files, sample_ids):
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