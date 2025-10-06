#!/usr/bin/env python3
"""
TAPESTRY Step 3: Interpret Components

Analyse learned components and identify cancer signatures.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.model.vae import BlindDeconvolutionVAE
from tapestry.interpret.analyser import ComponentAnalyser
from tapestry.utils.visualisations import (
    plot_component_methylation,
    plot_component_correlations,
    plot_proportion_distributions,
    plot_tf_correlation,
    plot_component_heatmap
)


def setup_logging(log_file='tapestry_interpret.log'):
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
        description='TAPESTRY: Interpret learned components'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/run_001/best_model.pt'),
        help='Path to trained model'
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
        default=Path('results/interpretation'),
        help='Output directory for interpretation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--ichorcna-file',
        type=Path,
        default=None,
        help='Path to ichorCNA tumor fraction CSV (columns: sample_name, TF)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/default_config.yaml'),
        help='Configuration file'
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'components').mkdir(exist_ok=True)
    (args.output_dir / 'regions').mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("TAPESTRY - Component Interpretation")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    
    methylation = np.load(args.data_dir / 'methylation_matrix.npy')
    coverage = np.load(args.data_dir / 'coverage_matrix.npy')
    regions = pd.read_csv(args.data_dir / 'regions.csv')
    sample_ids = pd.read_csv(args.data_dir / 'sample_ids.csv')['sample_id'].tolist()
    
    # Load ichorCNA tumor fractions if available
    if args.ichorcna_file and args.ichorcna_file.exists():
        ichorcna = pd.read_csv(args.ichorcna_file)
        # Rename columns to standard format
        ichorcna = ichorcna.rename(columns={'sample_name': 'sample_id', 'TF': 'ichorCNA_tf'})
        logger.info(f"Loaded ichorCNA data for {len(ichorcna)} samples")

        # Filter to only samples with ichorCNA data
        ichorcna_samples = set(ichorcna['sample_id'].values)
        missing_samples = [s for s in sample_ids if s not in ichorcna_samples]

        if missing_samples:
            logger.warning(f"Excluding {len(missing_samples)} samples without ichorCNA data:")
            for s in missing_samples:
                logger.warning(f"  - {s}")

        # Filter all data to only include samples with ichorCNA
        keep_indices = [i for i, s in enumerate(sample_ids) if s in ichorcna_samples]
        sample_ids = [sample_ids[i] for i in keep_indices]
        methylation = methylation[keep_indices]
        coverage = coverage[keep_indices]

        logger.info(f"Proceeding with {len(sample_ids)} samples with ichorCNA data")

        # Create metadata dataframe
        metadata = pd.DataFrame({'sample_id': sample_ids})
        metadata = metadata.merge(ichorcna[['sample_id', 'ichorCNA_tf']], on='sample_id', how='inner')
    else:
        metadata = None
        logger.warning("No ichorCNA file provided")
    
    # Load model
    logger.info("\nLoading trained model...")

    checkpoint = torch.load(args.model_path, map_location=args.device)

    model = BlindDeconvolutionVAE(
        n_regions=methylation.shape[1],
        n_components=config['model']['n_components'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    logger.info(f"Loaded model with {config['model']['n_components']} components")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'unknown'):.1f}")
    
    # Get proportions for all samples
    logger.info("\nComputing component proportions...")
    
    meth_tensor = torch.FloatTensor(methylation).to(args.device)
    cov_tensor = torch.FloatTensor(coverage).to(args.device)
    
    with torch.no_grad():
        proportions = model.get_proportions(meth_tensor, cov_tensor)
        proportions = proportions.cpu().numpy()
    
    # Get signatures
    signatures = model.signatures.cpu().numpy()
    
    # Initialise analyser
    logger.info("\nInitialising component analyser...")
    
    analyser = ComponentAnalyser(
        signatures=signatures,
        proportions=proportions,
        regions=regions,
        sample_ids=sample_ids,
        metadata=metadata
    )
    
    # Compute component statistics
    logger.info("\nComputing component statistics...")
    
    stats = analyser.compute_component_statistics()
    stats.to_csv(args.output_dir / 'component_statistics.csv', index=False)
    
    logger.info("\nComponent Statistics:")
    print(stats.to_string(index=False))
    
    # Identify cancer components
    logger.info("\nIdentifying cancer-related components...")
    
    cancer_components, cancer_analysis = analyser.identify_cancer_components()
    cancer_analysis.to_csv(args.output_dir / 'cancer_component_analysis.csv', index=False)
    
    if len(cancer_components) > 0:
        logger.info(f"\nCancer Components: {cancer_components}")
        print("\nCancer Component Analysis:")
        print(cancer_analysis[cancer_analysis['component'].isin(cancer_components)].to_string(index=False))
        
        # Save cancer component IDs
        pd.DataFrame({'cancer_component': cancer_components}).to_csv(
            args.output_dir / 'cancer_components.csv', index=False
        )
    else:
        logger.warning("No cancer components identified!")
    
    # Visualisations
    logger.info("\nGenerating visualisations...")
    
    # 1. Proportion distributions
    plot_proportion_distributions(
        proportions,
        save_path=args.output_dir / 'proportion_distributions.png'
    )
    
    # 2. Component heatmap
    plot_component_heatmap(
        proportions,
        sample_ids,
        save_path=args.output_dir / 'component_heatmap.png'
    )
    
    # 3. Correlations with metadata
    if metadata is not None:
        plot_component_correlations(
            proportions,
            metadata,
            save_path=args.output_dir / 'component_correlations.png'
        )
    
    # 4. Individual component methylation patterns
    for comp_idx in range(config['model']['n_components']):
        is_cancer = comp_idx in cancer_components
        suffix = '_CANCER' if is_cancer else ''
        
        plot_component_methylation(
            signatures,
            comp_idx,
            regions,
            save_path=args.output_dir / f'components/component_{comp_idx}{suffix}_methylation.png'
        )
    
    # 5. TF correlations for cancer components
    if metadata is not None and 'ichorCNA_tf' in metadata.columns and len(cancer_components) > 0:
        for comp_idx in cancer_components:
            plot_tf_correlation(
                proportions,
                metadata['ichorCNA_tf'].values,
                comp_idx,
                component_name=f'Cancer Component {comp_idx}',
                save_path=args.output_dir / f'components/component_{comp_idx}_tf_correlation.png'
            )
    
    # Export top regions for each cancer component
    logger.info("\nExporting top differentially methylated regions...")
    
    for comp_idx in cancer_components:
        top_regions = analyser.get_top_regions_for_component(comp_idx, n_top=100)
        top_regions.to_csv(
            args.output_dir / f'regions/component_{comp_idx}_top_regions.csv',
            index=False
        )
    
    # Compute cancer scores
    if len(cancer_components) > 0:
        logger.info("\nComputing sample cancer scores...")
        
        cancer_scores = analyser.compute_sample_cancer_scores(cancer_components)
        
        # Save scores
        score_df = pd.DataFrame({
            'sample_id': sample_ids,
            'cancer_score': cancer_scores
        })
        
        if metadata is not None:
            score_df = score_df.merge(
                metadata[['sample_id', 'diagnosis', 'ichorCNA_tf']] 
                if all(c in metadata.columns for c in ['diagnosis', 'ichorCNA_tf'])
                else metadata[['sample_id']],
                on='sample_id',
                how='left'
            )
        
        score_df.to_csv(args.output_dir / 'sample_cancer_scores.csv', index=False)
        
        # Evaluate performance
        logger.info("\nEvaluating cancer detection performance...")
        
        metrics = analyser.evaluate_performance(cancer_scores)
        
        if metrics:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(args.output_dir / 'performance_metrics.csv', index=False)
            
            logger.info("\nPerformance Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Interpretation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()