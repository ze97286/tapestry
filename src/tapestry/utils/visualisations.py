"""
Visualisation utilities for TAPESTRY.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def plot_component_methylation(
    signatures: np.ndarray,
    component_idx: int,
    regions: pd.DataFrame,
    top_k: int = 50,
    save_path: Optional[Path] = None
):
    """
    Plot methylation pattern for a specific component.
    
    Args:
        signatures: Signature matrix [n_components × n_regions]
        component_idx: Component to visualise
        regions: DataFrame with region annotations
        top_k: Number of top regions to show
        save_path: Path to save figure
    """
    component_meth = signatures[component_idx]
    
    # Get top hypermethylated and hypomethylated regions
    sorted_indices = np.argsort(component_meth)
    hypo_indices = sorted_indices[:top_k//2]  # Lowest methylation
    hyper_indices = sorted_indices[-top_k//2:]  # Highest methylation
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of all methylation values
    axes[0, 0].hist(component_meth, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(component_meth.mean(), color='red', 
                       linestyle='--', label=f'Mean: {component_meth.mean():.3f}')
    axes[0, 0].set_xlabel('Methylation Rate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Component {component_idx}: Methylation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top hypomethylated regions
    hypo_values = component_meth[hypo_indices]
    hypo_regions = regions.iloc[hypo_indices]
    
    y_pos = np.arange(len(hypo_values))
    axes[0, 1].barh(y_pos, hypo_values, color='blue', alpha=0.6)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels([
        f"{r['chrom']}:{r['start']//1000}k" 
        for _, r in hypo_regions.iterrows()
    ], fontsize=6)
    axes[0, 1].set_xlabel('Methylation Rate')
    axes[0, 1].set_title(f'Top {len(hypo_values)} Hypomethylated Regions')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Top hypermethylated regions
    hyper_values = component_meth[hyper_indices]
    hyper_regions = regions.iloc[hyper_indices]
    
    y_pos = np.arange(len(hyper_values))
    axes[1, 1].barh(y_pos, hyper_values, color='red', alpha=0.6)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([
        f"{r['chrom']}:{r['start']//1000}k" 
        for _, r in hyper_regions.iterrows()
    ], fontsize=6)
    axes[1, 1].set_xlabel('Methylation Rate')
    axes[1, 1].set_title(f'Top {len(hyper_values)} Hypermethylated Regions')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 4. Bimodality score
    low_meth = (component_meth < 0.2).mean()
    high_meth = (component_meth > 0.8).mean()
    mid_meth = ((component_meth >= 0.2) & (component_meth <= 0.8)).mean()
    
    categories = ['Low\n(<20%)', 'Medium\n(20-80%)', 'High\n(>80%)']
    values = [low_meth, mid_meth, high_meth]
    colours = ['blue', 'gray', 'red']
    
    axes[1, 0].bar(categories, values, color=colours, alpha=0.6, edgecolor='black')
    axes[1, 0].set_ylabel('Fraction of Regions')
    axes[1, 0].set_title('Methylation Distribution Categories')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add bimodality score
    bimodality = low_meth + high_meth
    axes[1, 0].text(
        0.5, 0.95, f'Bimodality Score: {bimodality:.3f}',
        transform=axes[1, 0].transAxes,
        ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.suptitle(f'Component {component_idx} Methylation Pattern', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved component methylation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_component_correlations(
    proportions: np.ndarray,
    metadata: pd.DataFrame,
    component_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot correlations between components and clinical variables.
    
    Args:
        proportions: Component proportions [n_samples × n_components]
        metadata: DataFrame with clinical variables
        component_names: Optional names for components
        save_path: Path to save figure
    """
    n_components = proportions.shape[1]
    
    if component_names is None:
        component_names = [f'C{i}' for i in range(n_components)]
    
    # Find numeric columns in metadata
    numeric_cols = metadata.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns in metadata for correlation")
        return
    
    # Calculate correlations
    correlations = []
    
    for col in numeric_cols:
        col_correlations = []
        for comp_idx in range(n_components):
            # Only correlate where both values are non-null
            valid_mask = ~np.isnan(metadata[col].values)
            if valid_mask.sum() < 10:
                col_correlations.append(0.0)
                continue
            
            corr = np.corrcoef(
                proportions[valid_mask, comp_idx],
                metadata[col].values[valid_mask]
            )[0, 1]
            col_correlations.append(corr)
        
        correlations.append(col_correlations)
    
    correlations = np.array(correlations)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(10, n_components), max(6, len(numeric_cols))))
    
    sns.heatmap(
        correlations,
        xticklabels=component_names,
        yticklabels=numeric_cols,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    
    ax.set_title('Component vs Clinical Variable Correlations', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Clinical Variable', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_proportion_distributions(
    proportions: np.ndarray,
    component_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot distribution of component proportions across samples.
    
    Args:
        proportions: Component proportions [n_samples × n_components]
        component_names: Optional names for components
        save_path: Path to save figure
    """
    n_components = proportions.shape[1]
    
    if component_names is None:
        component_names = [f'Component {i}' for i in range(n_components)]
    
    # Calculate statistics
    means = proportions.mean(axis=0)
    stds = proportions.std(axis=0)
    
    # Sort by mean proportion
    sorted_indices = np.argsort(means)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Violin plot
    data_for_violin = []
    labels_for_violin = []
    
    for idx in sorted_indices:
        data_for_violin.append(proportions[:, idx])
        labels_for_violin.append(component_names[idx])
    
    parts = axes[0].violinplot(
        data_for_violin,
        positions=range(n_components),
        showmeans=True,
        showmedians=True
    )
    
    axes[0].set_xticks(range(n_components))
    axes[0].set_xticklabels(labels_for_violin, rotation=45, ha='right')
    axes[0].set_ylabel('Proportion')
    axes[0].set_title('Component Proportion Distributions')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Colour violin plots by mean value
    for pc, mean_val in zip(parts['bodies'], means[sorted_indices]):
        pc.set_facecolor(plt.cm.viridis(mean_val / means.max()))
        pc.set_alpha(0.7)
    
    # 2. Bar plot with error bars
    x_pos = np.arange(n_components)
    axes[1].bar(
        x_pos,
        means[sorted_indices],
        yerr=stds[sorted_indices],
        capsize=5,
        alpha=0.7,
        color=[plt.cm.viridis(m / means.max()) for m in means[sorted_indices]],
        edgecolor='black'
    )
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_for_violin, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Proportion ± SD')
    axes[1].set_title('Mean Component Proportions')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved proportion distributions to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tf_correlation(
    proportions: np.ndarray,
    true_tf: np.ndarray,
    component_idx: int,
    component_name: str = None,
    save_path: Optional[Path] = None
):
    """
    Plot correlation between component proportion and tumour fraction.
    
    Args:
        proportions: Component proportions [n_samples × n_components]
        true_tf: Ground truth tumour fractions
        component_idx: Component to plot
        component_name: Optional name for component
        save_path: Path to save figure
    """
    component_props = proportions[:, component_idx]
    
    # Remove NaN values
    valid_mask = ~np.isnan(true_tf)
    component_props = component_props[valid_mask]
    true_tf_valid = true_tf[valid_mask]
    
    if len(component_props) < 5:
        logger.warning("Too few valid samples for TF correlation plot")
        return
    
    # Calculate correlation
    correlation = np.corrcoef(component_props, true_tf_valid)[0, 1]
    
    # Fit line
    z = np.polyfit(true_tf_valid, component_props, 1)
    p = np.poly1d(z)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        true_tf_valid,
        component_props,
        c=true_tf_valid,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    
    # Fit line
    x_line = np.linspace(true_tf_valid.min(), true_tf_valid.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
            label=f'Linear Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Diagonal (perfect correlation)
    max_val = max(true_tf_valid.max(), component_props.max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1, 
            alpha=0.5, label='Perfect Correlation')
    
    # Labels and title
    ax.set_xlabel('ichorCNA Tumour Fraction', fontsize=12)
    ax.set_ylabel(f'{component_name or f"Component {component_idx}"} Proportion', 
                  fontsize=12)
    ax.set_title(f'TF Correlation: r = {correlation:.3f}', 
                 fontsize=14, fontweight='bold')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Colourbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ichorCNA TF', rotation=270, labelpad=20)
    
    # Add text with statistics
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        true_tf_valid, component_props
    )
    
    textstr = f'r = {correlation:.3f}\np = {p_value:.2e}\nN = {len(component_props)}'
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved TF correlation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_composition(
    proportions: np.ndarray,
    sample_indices: List[int],
    sample_names: List[str],
    component_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot stacked bar chart of component composition for specific samples.
    
    Args:
        proportions: Component proportions [n_samples × n_components]
        sample_indices: Indices of samples to plot
        sample_names: Names for samples
        component_names: Optional names for components
        save_path: Path to save figure
    """
    n_components = proportions.shape[1]
    
    if component_names is None:
        component_names = [f'C{i}' for i in range(n_components)]
    
    # Extract proportions for selected samples
    sample_props = proportions[sample_indices]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(max(10, len(sample_indices) * 0.5), 6))
    
    # Generate colours
    colours = plt.cm.tab20(np.linspace(0, 1, n_components))
    
    # Plot stacked bars
    bottom = np.zeros(len(sample_indices))
    
    for comp_idx in range(n_components):
        values = sample_props[:, comp_idx]
        ax.bar(
            sample_names,
            values,
            bottom=bottom,
            label=component_names[comp_idx],
            color=colours[comp_idx],
            edgecolor='black',
            linewidth=0.5
        )
        bottom += values
    
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Sample Component Composition', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sample composition plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_component_heatmap(
    proportions: np.ndarray,
    sample_names: List[str],
    component_names: Optional[List[str]] = None,
    cluster_samples: bool = True,
    save_path: Optional[Path] = None
):
    """
    Plot heatmap of component proportions across all samples.
    
    Args:
        proportions: Component proportions [n_samples × n_components]
        sample_names: Names for samples
        component_names: Optional names for components
        cluster_samples: Whether to cluster samples by similarity
        save_path: Path to save figure
    """
    n_samples, n_components = proportions.shape
    
    if component_names is None:
        component_names = [f'Component {i}' for i in range(n_components)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_components), max(8, n_samples * 0.2)))
    
    # Plot heatmap
    sns.heatmap(
        proportions,
        xticklabels=component_names,
        yticklabels=sample_names,
        cmap='YlOrRd',
        vmin=0,
        vmax=proportions.max(),
        cbar_kws={'label': 'Proportion'},
        ax=ax,
        linewidths=0.5 if n_samples < 50 else 0
    )
    
    if cluster_samples and n_samples > 2:
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Compute linkage
        linkage_matrix = linkage(proportions, method='ward')
        
        # Reorder based on clustering
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']
        
        # Reorder heatmap
        ax.clear()
        sns.heatmap(
            proportions[order],
            xticklabels=component_names,
            yticklabels=[sample_names[i] for i in order],
            cmap='YlOrRd',
            vmin=0,
            vmax=proportions.max(),
            cbar_kws={'label': 'Proportion'},
            ax=ax,
            linewidths=0.5 if n_samples < 50 else 0
        )
    
    ax.set_title('Component Proportions Across Samples', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Sample', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved component heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()