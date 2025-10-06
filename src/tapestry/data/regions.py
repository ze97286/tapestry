"""
Define genomic regions for aggregation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class RegionDefiner:
    """
    Define genomic regions for methylation aggregation.
    """
    
    # Human chromosome sizes (hg38)
    CHROM_SIZES = {
        'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
        'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
        'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
        'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
        'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
        'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
        'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
        'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415,
    }
    
    def __init__(
        self,
        region_size: int = 500,
        region_step: int = 500,
        chromosomes: Optional[List[str]] = None
    ):
        """
        Initialize region definer.
        
        Args:
            region_size: Size of each region in bp
            region_step: Step size between regions (non-overlapping if equals region_size)
            chromosomes: List of chromosomes to include (default: chr1-22, X, Y)
        """
        self.region_size = region_size
        self.region_step = region_step
        
        if chromosomes is None:
            # Default: autosomal + sex chromosomes
            chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
        
        self.chromosomes = [c for c in chromosomes if c in self.CHROM_SIZES]
        
    def create_tiling_regions(self) -> pd.DataFrame:
        """
        Create tiling windows across the genome.
        
        Returns:
            DataFrame with columns: chrom, start, end, region_id
        """
        regions = []
        region_id = 0
        
        logger.info(
            f"Creating {self.region_size}bp tiling windows "
            f"with {self.region_step}bp steps..."
        )
        
        for chrom in self.chromosomes:
            chrom_size = self.CHROM_SIZES[chrom]
            
            # Create windows
            for start in range(0, chrom_size, self.region_step):
                end = min(start + self.region_size, chrom_size)
                
                regions.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'region_id': f'region_{region_id}',
                    'type': 'tiling'
                })
                
                region_id += 1
                
                if end >= chrom_size:
                    break
        
        df = pd.DataFrame(regions)
        
        logger.info(f"Created {len(df)} tiling regions")
        
        return df
    
    def load_promoter_regions(
        self,
        gtf_file: Optional[str] = None,
        upstream: int = 2000,
        downstream: int = 2000
    ) -> pd.DataFrame:
        """
        Load promoter regions (TSS ± window).
        
        Args:
            gtf_file: Path to GTF annotation file (if None, returns empty DataFrame)
            upstream: bp upstream of TSS
            downstream: bp downstream of TSS
            
        Returns:
            DataFrame with promoter regions
        """
        if gtf_file is None:
            logger.warning("No GTF file provided, skipping promoter regions")
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'region_id', 'type'])
        
        # TODO: Implement GTF parsing
        # For now, return empty
        logger.warning("Promoter loading not yet implemented")
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'region_id', 'type'])
    
    def load_cpg_islands(
        self,
        cpg_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load CpG island regions.
        
        Args:
            cpg_file: Path to CpG island BED file
            
        Returns:
            DataFrame with CpG island regions
        """
        if cpg_file is None:
            logger.warning("No CpG island file provided")
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'region_id', 'type'])
        
        # TODO: Implement CpG island loading
        logger.warning("CpG island loading not yet implemented")
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'region_id', 'type'])
    
    def create_hybrid_regions(
        self,
        gtf_file: Optional[str] = None,
        cpg_file: Optional[str] = None,
        promoter_upstream: int = 2000,
        promoter_downstream: int = 2000
    ) -> pd.DataFrame:
        """
        Create hybrid regions combining tiling + functional elements.
        
        Args:
            gtf_file: Path to GTF file for promoters
            cpg_file: Path to CpG island file
            promoter_upstream: bp upstream of TSS
            promoter_downstream: bp downstream of TSS
            
        Returns:
            Combined DataFrame of all regions
        """
        # Get tiling regions
        tiling = self.create_tiling_regions()
        
        # Get functional regions
        promoters = self.load_promoter_regions(
            gtf_file, promoter_upstream, promoter_downstream
        )
        cpg_islands = self.load_cpg_islands(cpg_file)
        
        # Combine
        all_regions = pd.concat([tiling, promoters, cpg_islands], ignore_index=True)
        
        # Remove duplicates (keep functional over tiling)
        # TODO: Implement overlap resolution
        
        logger.info(f"Created {len(all_regions)} total regions")
        
        return all_regions