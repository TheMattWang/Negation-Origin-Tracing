"""
Results interpretation tools for identifying negation layers and verifying causality.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_probe_results(results_path: str) -> pd.DataFrame:
    """
    Load probe results from JSON or CSV.
    
    Args:
        results_path: Path to results file
    
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


def load_intervention_results(results_path: str) -> Dict:
    """
    Load intervention results from JSON.
    
    Args:
        results_path: Path to intervention_results.json
    
    Returns:
        Dictionary with intervention results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def identify_negation_layers(
    probe_df: pd.DataFrame,
    intervention_results: Optional[Dict] = None,
    metric: str = "test_auroc",
    top_k: int = 3,
    min_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Identify the most important layers for negation based on probe and intervention results.
    
    Args:
        probe_df: DataFrame with probe results
        intervention_results: Optional intervention results dictionary
        metric: Metric to use for ranking ('test_accuracy', 'test_auroc')
        top_k: Number of top layers to return
        min_threshold: Minimum threshold for metric (optional)
    
    Returns:
        List of dictionaries with layer information, ranked by importance
    """
    # Start with probe results
    layer_scores = defaultdict(lambda: {
        'probe_scores': [],
        'pooling_strategies': [],
        'intervention_label_flips': [],
        'intervention_logit_deltas': []
    })
    
    # Aggregate probe scores by layer
    for _, row in probe_df.iterrows():
        layer_idx = row['layer_idx']
        layer_scores[layer_idx]['probe_scores'].append(row[metric])
        layer_scores[layer_idx]['pooling_strategies'].append(row['pooling_strategy'])
    
    # Add intervention results if available
    if intervention_results:
        for exp_type, layer_summary in intervention_results.items():
            for layer_idx_str, stats in layer_summary.items():
                layer_idx = int(layer_idx_str)
                if 'avg_label_flips' in stats:
                    layer_scores[layer_idx]['intervention_label_flips'].append(
                        stats['avg_label_flips']
                    )
                if 'avg_logit_delta' in stats:
                    layer_scores[layer_idx]['intervention_logit_deltas'].append(
                        stats['avg_logit_delta']
                    )
    
    # Compute composite scores
    ranked_layers = []
    for layer_idx, scores in layer_scores.items():
        # Average probe score across pooling strategies
        avg_probe_score = np.mean(scores['probe_scores'])
        
        # Average intervention effects
        avg_label_flips = np.mean(scores['intervention_label_flips']) if scores['intervention_label_flips'] else 0.0
        avg_logit_delta = np.mean(scores['intervention_logit_deltas']) if scores['intervention_logit_deltas'] else 0.0
        
        # Best pooling strategy for this layer
        best_pooling_idx = np.argmax(scores['probe_scores'])
        best_pooling = scores['pooling_strategies'][best_pooling_idx]
        best_probe_score = scores['probe_scores'][best_pooling_idx]
        
        # Composite score (weighted combination)
        # Higher probe score + higher intervention effects = more important
        composite_score = (
            0.6 * avg_probe_score +  # Probe performance (60%)
            0.2 * avg_label_flips +  # Intervention label flips (20%)
            0.2 * avg_logit_delta    # Intervention logit delta (20%)
        )
        
        ranked_layers.append({
            'layer_idx': layer_idx,
            'composite_score': composite_score,
            'avg_probe_score': avg_probe_score,
            'best_probe_score': best_probe_score,
            'best_pooling_strategy': best_pooling,
            'avg_label_flips': avg_label_flips,
            'avg_logit_delta': avg_logit_delta,
            'num_pooling_strategies': len(scores['pooling_strategies']),
        })
    
    # Sort by composite score
    ranked_layers.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Apply threshold if specified
    if min_threshold is not None:
        ranked_layers = [l for l in ranked_layers if l['avg_probe_score'] >= min_threshold]
    
    # Return top K
    return ranked_layers[:top_k]


def verify_causality(
    intervention_results: Dict,
    probe_results: pd.DataFrame,
    layer_idx: int,
    min_label_flip_rate: float = 0.1,
    min_logit_delta: float = 0.1
) -> Dict:
    """
    Verify if a layer shows causal effects based on intervention results.
    
    Args:
        intervention_results: Intervention results dictionary
        probe_results: Probe results DataFrame
        layer_idx: Layer to check
        min_label_flip_rate: Minimum label flip rate to consider causal
        min_logit_delta: Minimum logit delta to consider causal
    
    Returns:
        Dictionary with causality verification results
    """
    layer_str = str(layer_idx)
    
    # Check activation patching results
    ap_results = intervention_results.get('activation_patching', {})
    ap_layer = ap_results.get(layer_str, {})
    
    # Check ablation results
    ab_results = intervention_results.get('targeted_ablation', {})
    ab_layer = ab_results.get(layer_str, {})
    
    # Get probe performance for this layer
    layer_probes = probe_results[probe_results['layer_idx'] == layer_idx]
    best_probe = layer_probes.loc[layer_probes['test_auroc'].idxmax()] if len(layer_probes) > 0 else None
    
    # Determine causality
    ap_label_flips = ap_layer.get('avg_label_flips', 0.0)
    ap_logit_delta = ap_layer.get('avg_logit_delta', 0.0)
    ab_label_flips = ab_layer.get('avg_label_flips', 0.0)
    ab_logit_delta = ab_layer.get('avg_logit_delta', 0.0)
    
    # Criteria for causality:
    # 1. High probe performance (shows information is present)
    # 2. High label flip rate under activation patching (shows it affects predictions)
    # 3. Significant logit delta under ablation (shows it's necessary)
    
    has_high_probe = best_probe is not None and best_probe['test_auroc'] > 0.7
    has_activation_effect = ap_label_flips > min_label_flip_rate or ap_logit_delta > min_logit_delta
    has_ablation_effect = ab_label_flips > min_label_flip_rate or ab_logit_delta > min_logit_delta
    
    is_causal = has_high_probe and (has_activation_effect or has_ablation_effect)
    
    return {
        'layer_idx': layer_idx,
        'is_causal': is_causal,
        'probe_auroc': best_probe['test_auroc'] if best_probe is not None else 0.0,
        'activation_patching_label_flips': ap_label_flips,
        'activation_patching_logit_delta': ap_logit_delta,
        'ablation_label_flips': ab_label_flips,
        'ablation_logit_delta': ab_logit_delta,
        'has_high_probe': has_high_probe,
        'has_activation_effect': has_activation_effect,
        'has_ablation_effect': has_ablation_effect,
        'confidence': 'high' if (has_activation_effect and has_ablation_effect) else 'medium' if is_causal else 'low'
    }


def generate_interpretation_report(
    probe_results_path: str,
    intervention_results_path: Optional[str] = None,
    output_path: str = "experiments/interpretation_report.json",
    top_k: int = 5
) -> Dict:
    """
    Generate a comprehensive interpretation report.
    
    Args:
        probe_results_path: Path to probe results
        intervention_results_path: Optional path to intervention results
        output_path: Path to save report
        top_k: Number of top layers to analyze
    
    Returns:
        Dictionary with interpretation report
    """
    # Load results
    probe_df = load_probe_results(probe_results_path)
    intervention_results = None
    if intervention_results_path:
        intervention_results = load_intervention_results(intervention_results_path)
    
    # Identify negation layers
    negation_layers = identify_negation_layers(
        probe_df,
        intervention_results,
        top_k=top_k
    )
    
    # Verify causality for top layers
    causality_results = []
    if intervention_results:
        for layer_info in negation_layers:
            causality = verify_causality(
                intervention_results,
                probe_df,
                layer_info['layer_idx']
            )
            causality_results.append(causality)
    
    # Generate report
    report = {
        'summary': {
            'total_layers_tested': probe_df['layer_idx'].nunique(),
            'total_experiments': len(probe_df),
            'best_probe_accuracy': probe_df['test_accuracy'].max(),
            'best_probe_auroc': probe_df['test_auroc'].max(),
            'best_layer': int(probe_df.loc[probe_df['test_auroc'].idxmax(), 'layer_idx']),
            'best_pooling': probe_df.loc[probe_df['test_auroc'].idxmax(), 'pooling_strategy'],
        },
        'negation_layers': negation_layers,
        'causality_verification': causality_results,
        'recommendations': []
    }
    
    # Add recommendations
    if negation_layers:
        best_layer = negation_layers[0]
        report['recommendations'].append(
            f"Layer {best_layer['layer_idx']} appears to be the most important for negation "
            f"(composite score: {best_layer['composite_score']:.4f})"
        )
        report['recommendations'].append(
            f"Best pooling strategy for layer {best_layer['layer_idx']}: {best_layer['best_pooling_strategy']}"
        )
    
    if causality_results:
        causal_layers = [c for c in causality_results if c['is_causal']]
        if causal_layers:
            report['recommendations'].append(
                f"Found {len(causal_layers)} layer(s) with verified causal effects on negation"
            )
        else:
            report['recommendations'].append(
                "No layers showed strong causal effects. Consider running more intervention experiments."
            )
    
    # Save report
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Interpretation report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate interpretation report from probe and intervention results"
    )
    parser.add_argument(
        "--probe_results",
        type=str,
        required=True,
        help="Path to probe results (JSON or CSV)",
    )
    parser.add_argument(
        "--intervention_results",
        type=str,
        default=None,
        help="Path to intervention results (JSON, optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/interpretation_report.json",
        help="Output path for report",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top layers to analyze",
    )
    
    args = parser.parse_args()
    
    report = generate_interpretation_report(
        args.probe_results,
        args.intervention_results,
        args.output,
        args.top_k
    )
    
    # Print summary
    print("\n" + "="*60)
    print("INTERPRETATION SUMMARY")
    print("="*60)
    print(f"\nBest Layer: {report['summary']['best_layer']}")
    print(f"Best Probe AUROC: {report['summary']['best_probe_auroc']:.4f}")
    print(f"\nTop {args.top_k} Negation Layers:")
    for i, layer in enumerate(report['negation_layers'], 1):
        print(f"  {i}. Layer {layer['layer_idx']}: "
              f"score={layer['composite_score']:.4f}, "
              f"AUROC={layer['avg_probe_score']:.4f}, "
              f"best_pooling={layer['best_pooling_strategy']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

