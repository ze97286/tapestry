"""
Load TAPS methylation data from sample files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def _load_sample_worker(filepath: Path, min_coverage: int, context_filter: List[str]) -> pd.DataFrame:
    """
    Worker function for parallel sample loading.
    Must be at module level for multiprocessing.
    """
    # Define dtypes explicitly to avoid type inference overhead
    dtype_spec = {
        'chrom': 'category',
        'start': 'int32',
        'end': 'int32',
        'strand': 'category',
        'rate': 'float32',
        'unmod': 'int32',
        'mod': 'int32',
        'class': 'category',
        'context': 'category'
    }

    # Only load columns we need
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
        engine='c',
        low_memory=False
    )

    # Filter to desired context (e.g., CpG only)
    if context_filter:
        df = df[df['context'].isin(context_filter)]

    # Calculate coverage in-place
    df['coverage'] = df['unmod'] + df['mod']

    # Filter by minimum coverage
    df = df[df['coverage'] >= min_coverage]

    # Rename start to pos (in-place)
    df.rename(columns={'start': 'pos'}, inplace=True)

    # Drop columns we don't need anymore
    df.drop(columns=['unmod', 'context'], inplace=True)

    # Calculate rate
    df['rate'] = df['mod'] / df['coverage']

    # Convert chrom to category
    df['chrom'] = df['chrom'].astype('category')

    return df


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

        Expected format:
        #chr  start  end  name  beta_est  strand  unmod  mod  no_snp  snp  coverage  genotype  gt_p_score  gt_conf_score

        Args:
            filepath: Path to sample file

        Returns:
            DataFrame with columns: chrom, pos, mod, coverage, rate
        """
        try:
            # Define dtypes for the new format
            dtype_spec = {
                '#chr': 'category',
                'start': 'int32',
                'end': 'int32',
                'name': 'str',
                'beta_est': 'float32',
                'strand': 'category',
                'unmod': 'int32',
                'mod': 'int32',
                'no_snp': 'int32',
                'snp': 'int32',
                'coverage': 'int32',
                'genotype': 'category',
                'gt_p_score': 'int32',
                'gt_conf_score': 'int32'
            }

            # Only load columns we need (include 'end' to identify CpG properly)
            usecols = ['#chr', 'start', 'end', 'unmod', 'mod', 'coverage']

            # Read file (header starts with #chr)
            df = pd.read_csv(
                filepath,
                sep='\t',
                dtype=dtype_spec,
                usecols=usecols,
                engine='c'
            )

            # Rename #chr to chrom
            df.rename(columns={'#chr': 'chrom'}, inplace=True)

            # Drop 'end' column - don't need it
            df.drop(columns=['end'], inplace=True)

            # Sort by chrom and start for efficient grouping
            df = df.sort_values(['chrom', 'start']).reset_index(drop=True)

            # Group by chrom and the floor of start position (CpG pairs are at n, n+1)
            df['cpg_id'] = df['start'] // 2  # Integer division groups adjacent positions

            # Process each chromosome separately to avoid massive MultiIndex
            merged_chunks = []
            for chrom in df['chrom'].cat.categories:
                chrom_df = df[df['chrom'] == chrom].copy()

                # Merge strands: group by cpg_id within this chromosome
                merged = chrom_df.groupby('cpg_id', as_index=False).agg({
                    'start': 'min',  # Take minimum start as CpG position
                    'unmod': 'sum',
                    'mod': 'sum',
                    'coverage': 'sum'
                })
                merged['chrom'] = chrom
                merged_chunks.append(merged)

            # Combine all chromosomes
            df = pd.concat(merged_chunks, ignore_index=True)

            # Rename start to pos
            df.rename(columns={'start': 'pos'}, inplace=True)
            df.drop(columns=['cpg_id'], inplace=True)

            # Filter by minimum coverage (after merging strands)
            df = df[df['coverage'] >= self.min_coverage].copy()

            # Calculate rate
            df['rate'] = df['mod'] / df['coverage'].replace(0, 1)  # Avoid div by zero

            # Keep only necessary columns
            df = df[['chrom', 'pos', 'mod', 'coverage', 'rate']].copy()

            # Convert chrom to category
            df['chrom'] = df['chrom'].astype('category')

            logger.info(f"Loaded {len(df):,} CpG sites from {filepath.name}")

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

        WARNING: This is SERIAL loading. For parallel loading use iter_samples_parallel().

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

    def iter_samples_parallel(
        self,
        sample_files: List[Path],
        sample_ids: Optional[List[str]] = None,
        batch_size: int = 10,
        n_workers: int = 4
    ):
        """
        Generator that yields batches of samples loaded in parallel.

        Args:
            sample_files: List of paths to sample files
            sample_ids: Optional list of sample IDs (defaults to filenames)
            batch_size: Number of samples to load in parallel
            n_workers: Number of parallel workers

        Yields:
            List of (sample_id, DataFrame) tuples (batch)
        """
        if sample_ids is None:
            sample_ids = [f.stem for f in sample_files]

        if len(sample_ids) != len(sample_files):
            raise ValueError(
                f"Number of sample IDs ({len(sample_ids)}) must match "
                f"number of files ({len(sample_files)})"
            )

        # Process in batches
        n_samples = len(sample_files)

        with tqdm(total=n_samples, desc="Loading samples") as pbar:
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_files = sample_files[batch_start:batch_end]
                batch_ids = sample_ids[batch_start:batch_end]

                # Load batch in parallel
                batch_results = []

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all files in batch
                    futures = {}
                    for sample_id, filepath in zip(batch_ids, batch_files):
                        future = executor.submit(
                            _load_sample_worker,
                            filepath,
                            self.min_coverage,
                            self.context_filter
                        )
                        futures[future] = sample_id

                    # Collect results as they complete
                    for future in as_completed(futures):
                        sample_id = futures[future]
                        try:
                            df = future.result()
                            batch_results.append((sample_id, df))
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error loading {sample_id}: {e}")
                            raise

                # Yield entire batch
                yield batch_results

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