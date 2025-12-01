"""
Visualization tools for probe results and analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_layer_search_results(results_path: str) -> pd.DataFrame:
    """
    Load layer search results from JSON or CSV.
    
    Args:
        results_path: Path to results_summary.json or results_summary.csv
    
    Returns:
        DataFrame with results
    """
    if results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            results = json.load(f)
        df = pd.DataFrame(results)
    elif results_path.endswith('.csv'):
        df = pd.read_csv(results_path)
    else:
        raise ValueError(f"Unknown file format: {results_path}")
    
    return df


def plot_layer_performance(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "test_accuracy",
    title: Optional[str] = None
):
    """
    Plot probe performance by layer.
    
    Args:
        df: DataFrame with results
        output_path: Path to save plot
        metric: Metric to plot ('test_accuracy', 'test_auroc', etc.)
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by pooling strategy
    for pooling in df['pooling_strategy'].unique():
        df_pooling = df[df['pooling_strategy'] == pooling]
        df_pooling = df_pooling.sort_values('layer_idx')
        
        ax.plot(
            df_pooling['layer_idx'],
            df_pooling[metric],
            marker='o',
            label=f'{pooling} pooling',
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'Probe Performance by Layer ({metric})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot to {output_path}")


def plot_pooling_comparison(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "test_accuracy"
):
    """
    Plot comparison of pooling strategies across layers.
    
    Args:
        df: DataFrame with results
        output_path: Path to save plot
        metric: Metric to plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot for easier plotting
    pivot_df = df.pivot_table(
        index='layer_idx',
        columns='pooling_strategy',
        values=metric,
        aggfunc='mean'
    )
    
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    for i, pooling in enumerate(pivot_df.columns):
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            pivot_df[pooling],
            width,
            label=f'{pooling} pooling',
            alpha=0.8
        )
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Pooling Strategy Comparison by Layer ({metric})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot to {output_path}")


def plot_performance_heatmap(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "test_accuracy"
):
    """
    Plot heatmap of performance (layer × pooling strategy).
    
    Args:
        df: DataFrame with results
        output_path: Path to save plot
        metric: Metric to plot
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Create pivot table
    pivot_df = df.pivot_table(
        index='layer_idx',
        columns='pooling_strategy',
        values=metric,
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax
    )
    
    ax.set_title(f'Performance Heatmap: Layer × Pooling Strategy ({metric})', fontsize=14)
    ax.set_xlabel('Pooling Strategy', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot to {output_path}")


def plot_best_layers(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "test_accuracy",
    top_k: int = 5
):
    """
    Plot top K best performing layer/pooling combinations.
    
    Args:
        df: DataFrame with results
        output_path: Path to save plot
        metric: Metric to rank by
        top_k: Number of top results to show
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by metric and get top K
    df_sorted = df.nlargest(top_k, metric)
    
    # Create labels
    labels = [
        f"Layer {row['layer_idx']}\n{row['pooling_strategy']}"
        for _, row in df_sorted.iterrows()
    ]
    
    # Create bar plot
    bars = ax.barh(
        range(len(df_sorted)),
        df_sorted[metric],
        alpha=0.8
    )
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Top {top_k} Best Performing Probes ({metric})', fontsize=14)
    ax.invert_yaxis()  # Top to bottom
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
        ax.text(
            value + 0.01,
            i,
            f'{value:.4f}',
            va='center',
            fontsize=10
        )
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot to {output_path}")


def generate_all_visualizations(
    results_path: str,
    output_dir: str,
    metrics: Optional[List[str]] = None
):
    """
    Generate all standard visualizations from layer search results.
    
    Args:
        results_path: Path to results_summary.json or .csv
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: ['test_accuracy', 'test_auroc'])
    """
    if metrics is None:
        metrics = ['test_accuracy', 'test_auroc']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = load_layer_search_results(results_path)
    
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠ Warning: Metric '{metric}' not found in results, skipping")
            continue
        
        # Layer performance plot
        plot_layer_performance(
            df,
            os.path.join(output_dir, f"layer_performance_{metric}.png"),
            metric=metric
        )
        
        # Pooling comparison
        plot_pooling_comparison(
            df,
            os.path.join(output_dir, f"pooling_comparison_{metric}.png"),
            metric=metric
        )
        
        # Heatmap
        plot_performance_heatmap(
            df,
            os.path.join(output_dir, f"heatmap_{metric}.png"),
            metric=metric
        )
        
        # Best layers
        plot_best_layers(
            df,
            os.path.join(output_dir, f"best_layers_{metric}.png"),
            metric=metric,
            top_k=5
        )
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate visualizations from layer search results"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to results_summary.json or .csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results_path)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="test_accuracy,test_auroc",
        help="Comma-separated list of metrics to plot",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or os.path.dirname(args.results_path)
    metrics = [m.strip() for m in args.metrics.split(",")]
    
    generate_all_visualizations(args.results_path, output_dir, metrics)

