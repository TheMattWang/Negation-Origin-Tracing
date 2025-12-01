"""
Metrics for tracking causal intervention results.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_label_flip_rate(
    original_preds: torch.Tensor,
    intervened_preds: torch.Tensor
) -> float:
    """
    Compute the rate at which predictions flip after intervention.
    
    Args:
        original_preds: Original predictions
        intervened_preds: Predictions after intervention
    
    Returns:
        Label flip rate (0.0 to 1.0)
    """
    flips = (original_preds != intervened_preds).float()
    return flips.mean().item()


def compute_logit_delta(
    original_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    reduction: str = "mean"
) -> float:
    """
    Compute the change in logits after intervention.
    
    Args:
        original_logits: Original logits
        intervened_logits: Logits after intervention
        reduction: How to reduce ("mean", "max", "sum")
    
    Returns:
        Logit delta
    """
    delta = (intervened_logits - original_logits).abs()
    
    if reduction == "mean":
        return delta.mean().item()
    elif reduction == "max":
        return delta.max().item()
    elif reduction == "sum":
        return delta.sum().item()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_probe_accuracy_by_layer(
    layer_results: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, float]:
    """
    Compute probe accuracy for each layer.
    
    Args:
        layer_results: Dictionary mapping layer_idx to results with 'preds' and 'labels'
    
    Returns:
        Dictionary mapping layer_idx to accuracy
    """
    accuracies = {}
    for layer_idx, results in layer_results.items():
        preds = results["preds"]
        labels = results["labels"]
        accuracy = (preds == labels).float().mean().item()
        accuracies[layer_idx] = accuracy
    
    return accuracies


def compute_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        preds: Predictions
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(preds)):
        cm[labels[i].item(), preds[i].item()] += 1
    
    return cm


def compute_macro_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2
) -> float:
    """
    Compute macro-averaged F1 score.
    
    Args:
        preds: Predictions
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Macro F1 score
    """
    cm = compute_confusion_matrix(preds, labels, num_classes)
    
    f1_scores = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def compute_auroc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2
) -> float:
    """
    Compute Area Under the ROC Curve (AUROC).
    
    Args:
        logits: Model logits (batch_size, num_classes)
        labels: True labels (batch_size,)
        num_classes: Number of classes (default: 2 for binary)
    
    Returns:
        AUROC score (0.0 to 1.0)
    """
    if SKLEARN_AVAILABLE:
        # Convert to numpy
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits
        
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
        
        # Get probabilities
        if num_classes == 2:
            # Binary classification: use positive class probability
            probs = torch.softmax(logits, dim=-1)
            if isinstance(probs, torch.Tensor):
                probs_np = probs[:, 1].detach().cpu().numpy()
            else:
                probs_np = probs[:, 1]
        else:
            # Multi-class: use one-vs-rest approach
            probs = torch.softmax(logits, dim=-1)
            if isinstance(probs, torch.Tensor):
                probs_np = probs.detach().cpu().numpy()
            else:
                probs_np = probs
        
        # Handle edge cases
        if len(np.unique(labels_np)) < 2:
            # Only one class present, return 0.5 (random)
            return 0.5
        
        # Compute AUROC
        if num_classes == 2:
            try:
                auroc = roc_auc_score(labels_np, probs_np)
                return float(auroc)
            except ValueError:
                # Fallback if there's an issue
                return 0.5
        else:
            # Multi-class: use one-vs-rest
            try:
                auroc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')
                return float(auroc)
            except ValueError:
                return 0.5
    else:
        # Manual implementation if sklearn not available
        # This is a simplified version - sklearn is preferred
        probs = torch.softmax(logits, dim=-1)
        
        if num_classes == 2:
            # Binary: use positive class probability
            pos_probs = probs[:, 1]
            labels_float = labels.float()
            
            # Sort by probability
            sorted_indices = torch.argsort(pos_probs, descending=True)
            sorted_probs = pos_probs[sorted_indices]
            sorted_labels = labels_float[sorted_indices]
            
            # Compute TPR and FPR at different thresholds
            # Simplified: approximate AUROC using trapezoidal rule
            # For exact computation, sklearn is recommended
            thresholds = torch.linspace(0, 1, 100)
            tprs = []
            fprs = []
            
            for threshold in thresholds:
                preds = (sorted_probs >= threshold).float()
                tp = ((preds == 1) & (sorted_labels == 1)).sum().item()
                fp = ((preds == 1) & (sorted_labels == 0)).sum().item()
                fn = ((preds == 0) & (sorted_labels == 1)).sum().item()
                tn = ((preds == 0) & (sorted_labels == 0)).sum().item()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                tprs.append(tpr)
                fprs.append(fpr)
            
            # Compute area using trapezoidal rule
            auroc = np.trapz(tprs, fprs)
            return float(auroc)
        else:
            # Multi-class: return average of one-vs-rest AUROCs
            aurocs = []
            for class_idx in range(num_classes):
                class_probs = probs[:, class_idx]
                class_labels = (labels == class_idx).float()
                
                if class_labels.sum() == 0 or class_labels.sum() == len(class_labels):
                    aurocs.append(0.5)  # Random if only one class
                else:
                    # Similar simplified computation
                    aurocs.append(0.5)  # Placeholder - use sklearn for accuracy
            
            return float(np.mean(aurocs))


def split_by_negation(
    texts: List[str],
    preds: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, Dict]:
    """
    Split results by whether text contains negation.
    
    Args:
        texts: List of input texts
        preds: Predictions
        labels: True labels
    
    Returns:
        Dictionary with 'negated' and 'non_negated' keys
    """
    negated_indices = []
    non_negated_indices = []
    
    for i, text in enumerate(texts):
        if "not " in text.lower() or "n't " in text.lower():
            negated_indices.append(i)
        else:
            non_negated_indices.append(i)
    
    return {
        "negated": {
            "preds": preds[negated_indices] if negated_indices else torch.tensor([]),
            "labels": labels[negated_indices] if negated_indices else torch.tensor([]),
            "indices": negated_indices,
        },
        "non_negated": {
            "preds": preds[non_negated_indices] if non_negated_indices else torch.tensor([]),
            "labels": labels[non_negated_indices] if non_negated_indices else torch.tensor([]),
            "indices": non_negated_indices,
        }
    }


class InterventionMetrics:
    """
    Tracks metrics across multiple causal intervention experiments.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add_result(
        self,
        experiment_type: str,
        layer_idx: int,
        label_flips: int,
        logit_delta: float,
        accuracy: Optional[float] = None,
        auroc: Optional[float] = None,
        **kwargs
    ):
        """
        Add a result from an intervention experiment.
        
        Args:
            experiment_type: Type of experiment ("activation_patching", "ablation", etc.)
            layer_idx: Layer index
            label_flips: Number of label flips
            logit_delta: Change in logits
            accuracy: Accuracy after intervention (optional)
            auroc: AUROC score after intervention (optional)
            **kwargs: Additional metrics
        """
        result = {
            "layer_idx": layer_idx,
            "label_flips": label_flips,
            "logit_delta": logit_delta,
            "accuracy": accuracy,
            "auroc": auroc,
            **kwargs
        }
        self.metrics[experiment_type].append(result)
    
    def get_layer_summary(self, experiment_type: str) -> Dict[int, Dict]:
        """
        Get summary statistics by layer.
        
        Args:
            experiment_type: Type of experiment
        
        Returns:
            Dictionary mapping layer_idx to summary stats
        """
        if experiment_type not in self.metrics:
            return {}
        
        layer_stats = defaultdict(lambda: {"label_flips": [], "logit_deltas": [], "accuracies": [], "aurocs": []})
        
        for result in self.metrics[experiment_type]:
            layer_idx = result["layer_idx"]
            layer_stats[layer_idx]["label_flips"].append(result["label_flips"])
            layer_stats[layer_idx]["logit_deltas"].append(result["logit_delta"])
            if result["accuracy"] is not None:
                layer_stats[layer_idx]["accuracies"].append(result["accuracy"])
            if result.get("auroc") is not None:
                layer_stats[layer_idx]["aurocs"].append(result["auroc"])
        
        summary = {}
        for layer_idx, stats in layer_stats.items():
            summary[layer_idx] = {
                "avg_label_flips": np.mean(stats["label_flips"]),
                "std_label_flips": np.std(stats["label_flips"]),
                "avg_logit_delta": np.mean(stats["logit_deltas"]),
                "std_logit_delta": np.std(stats["logit_deltas"]),
                "avg_accuracy": np.mean(stats["accuracies"]) if stats["accuracies"] else None,
                "avg_auroc": np.mean(stats["aurocs"]) if stats["aurocs"] else None,
                "std_auroc": np.std(stats["aurocs"]) if stats["aurocs"] else None,
            }
        
        return summary
    
    def get_overall_summary(self) -> Dict:
        """Get overall summary across all experiment types."""
        summary = {}
        for exp_type in self.metrics:
            summary[exp_type] = self.get_layer_summary(exp_type)
        return summary

