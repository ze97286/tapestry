#!/usr/bin/env python3
"""
TAPESTRY Step 4: Predict on New Cohort

Process raw TAPS samples and estimate tumour fraction.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.data.loader import TAPSLoader
from tapestry.data.aggregator import RegionAggregator
from tapestry.model.vae import BlindDeconvolutionVAE


def setup_logging(log_file='tapestry_predict.log'):
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Predict tumour fraction for new samples'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing new TAPS sample files'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/run_001/best_model.pt'),
        help='Path to trained model'
    )
    parser.add_argument(
        '--regions-path',
        type=Path,
        default=Path('data/processed/regions.csv'),
        help='Path to regions CSV (from training)'
    )
    parser.add_argument(
        '--cancer-components-path',
        type=Path,
        default=Path('results/interpretation/cancer_components.csv'),
        help='Path to cancer components CSV'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('predictions.csv'),
        help='Output file for predictions'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.bed',
        help='File pattern to match'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("TAPESTRY - Tumour Fraction Prediction")
    logger.info("="*60)
    
    # Load model
    logger.info("\nLoading trained model...")
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint['config']
    
    model = BlindDeconvolutionVAE(
        n_regions=checkpoint['n_regions'],
        n_components=config['model']['n_components'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    logger.info(f"Loaded model with {config['model']['n_components']} components")
    
    # Load regions
    logger.info("\nLoading regions...")
    regions = pd.read_csv(args.regions_path)
    logger.info(f"Loaded {len(regions)} regions")
    
    # Load cancer components
    logger.info("\nLoading cancer components...")
    cancer_df = pd.read_csv(args.cancer_components_path)
    cancer_components = cancer_df['cancer_component'].tolist()
    logger.info(f"Cancer components: {cancer_components}")
    
    # Load new samples
    logger.info("\nLoading new samples...")
    
    sample_files = sorted(args.input_dir.glob(args.pattern))
    
    if len(sample_files) == 0:
        logger.error(f"No files found matching {args.pattern} in {args.input_dir}")
        return
    
    logger.info(f"Found {len(sample_files)} sample files")
    
    loader = TAPSLoader(
        min_coverage=config['data']['min_cpg_coverage'],
        context_filter=config['data']['cpg_context_filter']
    )
    
    cohort_data = loader.load_cohort(sample_files)
    
    # Aggregate to regions
    logger.info("\nAggregating CpGs to regions...")
    
    aggregator = RegionAggregator(
        min_cpgs_per_region=config['filtering']['min_cpgs_per_region'],
        min_region_coverage=config['data']['min_region_coverage']
    )
    
    # Process each sample
    results = []
    
    for sample_id, cpg_data in tqdm(cohort_data.items(), desc="Processing samples"):
        # Aggregate this sample
        sample_regions = aggregator.aggregate_sample(cpg_data, regions)
        
        # Create matrices aligned with training regions
        meth_vector = np.zeros(len(regions))
        cov_vector = np.zeros(len(regions))
        
        # Map aggregated regions to training region indices
        for _, row in sample_regions.iterrows():
            region_mask = (
                (regions['chrom'] == row['chrom']) &
                (regions['start'] == row['start']) &
                (regions['end'] == row['end'])
            )
            
            indices = np.where(region_mask)[0]
            
            if len(indices) > 0:
                idx = indices[0]
                meth_vector[idx] = row['mod_count']
                cov_vector[idx] = row['coverage']
        
        # Check coverage
        n_covered = (cov_vector > 0).sum()
        coverage_fraction = n_covered / len(regions)
        
        if coverage_fraction < 0.1:  # Less than 10% of regions covered
            logger.warning(
                f"Sample {sample_id} has low coverage "
                f"({coverage_fraction:.1%} of regions)"
            )
        
        # Predict
        meth_tensor = torch.FloatTensor(meth_vector).unsqueeze(0).to(args.device)
        cov_tensor = torch.FloatTensor(cov_vector).unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            proportions = model.get_proportions(meth_tensor, cov_tensor)
            proportions = proportions.cpu().numpy()[0]
        
        # Compute cancer score
        cancer_score = proportions[cancer_components].sum()
        
        # Store results
        result = {
            'sample_id': sample_id,
            'predicted_tf': cancer_score,
            'n_regions_covered': n_covered,
            'coverage_fraction': coverage_fraction,
        }
        
        # Add individual component proportions
        for comp_idx in range(len(proportions)):
            is_cancer = comp_idx in cancer_components
            comp_name = f'component_{comp_idx}{"_CANCER" if is_cancer else ""}'
            result[comp_name] = proportions[comp_idx]
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by predicted TF (descending)
    results_df = results_df.sort_values('predicted_tf', ascending=False)
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("Prediction Summary")
    logger.info("="*60)
    logger.info(f"\nProcessed {len(results_df)} samples")
    logger.info(f"\nPredicted TF statistics:")
    logger.info(f"  Mean: {results_df['predicted_tf'].mean():.4f}")
    logger.info(f"  Median: {results_df['predicted_tf'].median():.4f}")
    logger.info(f"  Min: {results_df['predicted_tf'].min():.4f}")
    logger.info(f"  Max: {results_df['predicted_tf'].max():.4f}")
    
    # Show top 10 samples
    logger.info(f"\nTop 10 samples by predicted TF:")
    print(results_df[['sample_id', 'predicted_tf', 'coverage_fraction']].head(10).to_string(index=False))
    
    logger.info(f"\nResults saved to: {args.output_file}")
    logger.info("="*60)


if __name__ == '__main__':
    main()