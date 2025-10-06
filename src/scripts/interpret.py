#!/usr/bin/env python3
"""
TAPESTRY Step 3: Interpret Components

Analyze learned components and validate model performance.
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
from tapestry.interpret.analyser import ComponentAnalyzer


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
        required=True,
        help='Path to trained model (.pt file)'
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
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--min-auc',
        type=float,
        default=0.70,
        help='Minimum AUC to consider component cancer-related'
    )
    parser.add_argument(
        '--min-correlation',
        type=float,
        default=0.50,
        help='Minimum TF correlation to consider component cancer-related'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TAPESTRY - Component Interpretation")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    
    methylation = np.load(args.data_dir / 'methylation_matrix.npy')
    coverage = np.load(args.data_dir / 'coverage_matrix.npy')
    regions = pd.read_csv(args.data_dir / 'regions.csv')
    sample_ids = pd.read_csv(args.data_dir / 'sample_ids.csv')['sample_id'].tolist()
    
    # Load metadata if available
    metadata_path = args.data_dir / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with columns: {metadata.columns.tolist()}")
    else:
        metadata = None
        logger.warning("No metadata found")
    
    # Load model
    logger.info("\nLoading trained model...")
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Reconstruct model
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Load from default config
        with open('config/default_config.yaml') as f:
            config = yaml.safe_load(f)
    
    model = BlindDeconvolutionVAE(
        n_regions=methylation.shape[1],
        n_components=config['model']['n_components'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model with {config['model']['n_components']} components")
    
    # Initialize analyzer
    logger.info("\nInitializing analyzer...")
    
    analyzer = ComponentAnalyzer(
        model=model,
        methylation=methylation,
        coverage=coverage,
        regions=regions,
        sample_ids=sample_ids,
        metadata=metadata,
        device=args.device
    )
    
    # Identify cancer components
    logger.info("\n" + "="*60)
    logger.info("COMPONENT INTERPRETATION")
    logger.info("="*60)
    
    component_interpretations = analyzer.identify_cancer_components(
        min_auc=args.min_auc,
        min_correlation=args.min_correlation
    )
    
    cancer_components = [
        idx for idx, info in component_interpretations.items()
        if info['is_cancer']
    ]
    
    if not cancer_components:
        logger.warning("\n⚠️  No cancer components identified!")
        logger.warning("Consider:")
        logger.warning("  - Lowering --min-auc or --min-correlation thresholds")
        logger.warning("  - Checking if metadata contains diagnosis/TF columns")
        logger.warning("  - Reviewing model training (may need more epochs)")
    else:
        logger.info(f"\n✓ Identified {len(cancer_components)} cancer components")
        
        # Detailed analysis of cancer components
        logger.info("\nDetailed Cancer Component Analysis:")
        for comp_idx in cancer_components:
            logger.info(f"\n  Component {comp_idx}:")
            info = component_interpretations[comp_idx]
            logger.info(f"    Mean proportion: {info['mean_proportion']:.4f}")
            logger.info(f"    Std proportion: {info['std_proportion']:.4f}")
            if 'auc' in info:
                logger.info(f"    AUC: {info['auc']:.3f}")
            if 'tf_correlation' in info:
                logger.info(f"    TF correlation: {info['tf_correlation']:.3f}")
            logger.info(f"    Evidence: {', '.join(info['evidence'])}")
            
            # Signature analysis
            sig_analysis = analyzer.analyze_signatures(comp_idx, top_k=10)
            logger.info(f"    Mean methylation: {sig_analysis['mean_methylation']:.3f}")
            logger.info(f"    Bimodality score: {sig_analysis['bimodality_score']:.3f}")
    
    # Validation (if we have test set)
    if metadata is not None and 'split' in metadata.columns:
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE VALIDATION")
        logger.info("="*60)
        
        test_indices = np.where(metadata['split'] == 'test')[0]
        
        if len(test_indices) > 0 and cancer_components:
            metrics = analyzer.validate_performance(
                test_indices,
                cancer_components
            )
        else:
            logger.warning("No test set found or no cancer components")
            metrics = {}
    else:
        # Use last 15% as test set
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE VALIDATION (using last 15% as test)")
        logger.info("="*60)
        
        n_test = int(len(sample_ids) * 0.15)
        test_indices = np.arange(len(sample_ids) - n_test, len(sample_ids))
        
        if cancer_components:
            metrics = analyzer.validate_performance(
                test_indices,
                cancer_components
            )
        else:
            metrics = {}
    
    # Generate plots
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    # Component summary plot
    logger.info("\nCreating component summary plot...")
    analyzer.plot_component_summary(
        component_interpretations,
        save_path=args.output_dir / 'component_summary.png'
    )
    
    # Calibration plot (if we have TF data)
    if metadata is not None and 'ichorCNA_tf' in metadata.columns and cancer_components:
        logger.info("Creating calibration plot...")
        
        # All samples
        analyzer.plot_calibration(
            cancer_components,
            test_indices=None,
            save_path=args.output_dir / 'calibration_all.png'
        )
        
        # Test set only
        if len(test_indices) > 0:
            analyzer.plot_calibration(
                cancer_components,
                test_indices=test_indices,
                save_path=args.output_dir / 'calibration_test.png'
            )
    
    # Export results
    logger.info("\n" + "="*60)
    logger.info("EXPORTING RESULTS")
    logger.info("="*60)
    
    analyzer.export_results(
        component_interpretations,
        cancer_components,
        args.output_dir
    )
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print("TAPESTRY Analysis Complete!")
    print("="*60)
    print(f"\nModel: {args.model_path.name}")
    print(f"Samples: {len(sample_ids)}")
    print(f"Regions: {len(regions)}")
    print(f"Components: {model.n_components}")
    print(f"\nCancer-related components: {cancer_components}")
    
    if metrics:
        print(f"\nTest Set Performance:")
        if 'auc' in metrics:
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Sensitivity: {metrics.get('sensitivity', 0):.3f}")
            print(f"  Specificity: {metrics.get('specificity', 0):.3f}")
        if 'tf_correlation_reliable' in metrics:
            print(f"  TF Correlation: {metrics['tf_correlation_reliable']:.3f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R²: {metrics['r2']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()