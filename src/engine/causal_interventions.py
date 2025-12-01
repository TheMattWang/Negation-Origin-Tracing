"""
Causal Intervention Module for Negation-Origin-Tracing.

Implements:
1. Activation patching: Swap hidden states between negated/non-negated pairs
2. Targeted ablations: Zero out or project out probe-identified dimensions
3. Control experiments: Label-shuffled, representation-shuffled, random interventions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from transformers import AutoModel


class ActivationPatcher:
    """
    Performs activation patching by swapping hidden states between pairs.
    """
    
    def __init__(self, model: AutoModel):
        """
        Args:
            model: The base model (frozen) to extract activations from
        """
        self.model = model
        self.hook_handles = []
        self.activations = {}
        self.patched_activations = {}
    
    def _get_activation_hook(self, layer_idx: int, position: Optional[int] = None):
        """Create a hook to capture activations."""
        def hook(module, input, output):
            # output is a tuple: (hidden_states, ...)
            # For DistilBERT, output[0] is the hidden states
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if position is not None:
                # Store specific token position
                self.activations[f"layer_{layer_idx}_pos_{position}"] = hidden_states[:, position, :].clone()
            else:
                # Store all positions
                self.activations[f"layer_{layer_idx}"] = hidden_states.clone()
        
        return hook
    
    def _get_patch_hook(self, layer_idx: int, patch_activations: torch.Tensor, position: Optional[int] = None):
        """Create a hook to patch activations."""
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if position is not None:
                # Patch specific position
                hidden_states[:, position, :] = patch_activations
            else:
                # Patch all positions
                hidden_states = patch_activations.clone()
            
            # Return modified output
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return hook
    
    def _get_encoder_layer(self, layer_idx: int):
        """
        Get the underlying encoder layer for a variety of HF model wrappers
        (plain encoders and classification models).
        """
        m = self.model
        # DistilBERT base model or similar
        if hasattr(m, "transformer"):
            return m.transformer.layer[layer_idx]
        # BERT / RoBERTa style encoders
        if hasattr(m, "encoder"):
            return m.encoder.layer[layer_idx]
        # DistilBERT *classification* model: distilbert.transformer.layer
        if hasattr(m, "distilbert") and hasattr(m.distilbert, "transformer"):
            return m.distilbert.transformer.layer[layer_idx]
        # BERT classification model: bert.encoder.layer
        if hasattr(m, "bert") and hasattr(m.bert, "encoder"):
            return m.bert.encoder.layer[layer_idx]
        # RoBERTa classification model: roberta.encoder.layer
        if hasattr(m, "roberta") and hasattr(m.roberta, "encoder"):
            return m.roberta.encoder.layer[layer_idx]

        raise ValueError("Could not find transformer/encoder layers in model")

    def register_hooks(self, layer_idx: int, position: Optional[int] = None):
        """Register hooks to capture activations at a specific layer."""
        layer = self._get_encoder_layer(layer_idx)
        
        # Register forward hook
        handle = layer.register_forward_hook(
            self._get_activation_hook(layer_idx, position)
        )
        self.hook_handles.append(handle)
        return handle
    
    def patch_activations(
        self,
        source_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        layer_idx: int,
        position: Optional[int] = None,
        return_logits: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Patch activations from source into target at specified layer/position.
        
        Args:
            source_batch: Batch to extract activations from
            target_batch: Batch to patch activations into
            layer_idx: Layer index to patch at
            position: Token position to patch (None = all positions)
            return_logits: Whether to return logits from patched model
        
        Returns:
            Dictionary with patched logits and/or activations
        """
        # Clear previous activations
        self.activations.clear()
        self.patched_activations.clear()
        
        # Step 1: Extract activations from source
        self.register_hooks(layer_idx, position)
        with torch.no_grad():
            _ = self.model(
                input_ids=source_batch["input_ids"],
                attention_mask=source_batch["attention_mask"],
                output_hidden_states=True
            )
        
        # Get the captured activations
        if position is not None:
            source_activations = self.activations[f"layer_{layer_idx}_pos_{position}"]
        else:
            source_activations = self.activations[f"layer_{layer_idx}"]
        
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # Step 2: Patch activations into target
        layer = self._get_encoder_layer(layer_idx)
        
        # Create patch hook
        def patch_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if position is not None:
                # Patch specific position
                hidden_states[:, position, :] = source_activations
            else:
                # Patch all positions (need to match batch size)
                if hidden_states.shape[0] == source_activations.shape[0]:
                    hidden_states = source_activations.clone()
                else:
                    # Broadcast if needed
                    hidden_states = source_activations.expand_as(hidden_states).clone()
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        handle = layer.register_forward_hook(patch_hook)
        
        # Step 3: Forward pass with patched activations
        with torch.no_grad():
            outputs = self.model(
                input_ids=target_batch["input_ids"],
                attention_mask=target_batch["attention_mask"],
                output_hidden_states=True
            )
        
        # Remove hook
        handle.remove()
        
        result = {}
        if return_logits:
            # Store patched hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                result["patched_hidden_states"] = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                result["patched_hidden_states"] = outputs.last_hidden_state
            else:
                result["patched_hidden_states"] = None
        
        return result


class TargetedAblation:
    """
    Performs targeted ablations by zeroing out or projecting out specific dimensions.
    """
    
    def __init__(self, model: AutoModel):
        """
        Args:
            model: The base model (frozen)
        """
        self.model = model
        self.hook_handles = []
    
    def zero_out_dimensions(
        self,
        batch: Dict[str, torch.Tensor],
        layer_idx: int,
        dimensions: Union[List[int], torch.Tensor],
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Zero out specific dimensions in hidden states.
        
        Args:
            batch: Input batch
            layer_idx: Layer to ablate
            dimensions: List of dimension indices to zero out
            position: Token position (None = all positions)
        
        Returns:
            Ablated hidden states
        """
        if isinstance(dimensions, list):
            dimensions = torch.tensor(dimensions, dtype=torch.long)
        
        # Get the transformer layer
        layer = self._get_encoder_layer(layer_idx)
        
        def ablation_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if position is not None:
                # Ablate specific position
                hidden_states[:, position, dimensions] = 0.0
            else:
                # Ablate all positions
                hidden_states[:, :, dimensions] = 0.0
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        handle = layer.register_forward_hook(ablation_hook)
        self.hook_handles.append(handle)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )
        
        handle.remove()
        
        return outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
    
    def project_out_dimensions(
        self,
        batch: Dict[str, torch.Tensor],
        layer_idx: int,
        projection_matrix: torch.Tensor,
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Project out specific dimensions using a projection matrix.
        
        Args:
            batch: Input batch
            layer_idx: Layer to ablate
            projection_matrix: Projection matrix (I - P) where P projects onto dimensions to remove
            position: Token position (None = all positions)
        
        Returns:
            Ablated hidden states
        """
        # Get the transformer layer
        layer = self._get_encoder_layer(layer_idx)
        
        def projection_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if position is not None:
                # Project specific position
                hidden_states[:, position, :] = torch.matmul(
                    hidden_states[:, position, :],
                    projection_matrix
                )
            else:
                # Project all positions
                batch_size, seq_len, hidden_size = hidden_states.shape
                hidden_states = hidden_states.view(-1, hidden_size)
                hidden_states = torch.matmul(hidden_states, projection_matrix)
                hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        handle = layer.register_forward_hook(projection_hook)
        self.hook_handles.append(handle)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )
        
        handle.remove()
        
        return outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None


class CausalInterventionRunner:
    """
    Main class for running causal interventions and tracking metrics.
    """
    
    def __init__(
        self,
        model: AutoModel,
        probe: Optional[nn.Module] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: Base model (frozen)
            probe: Optional trained probe to use for predictions
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.probe = probe.to(device) if probe else None
        self.device = device
        
        self.patcher = ActivationPatcher(self.model)
        self.ablator = TargetedAblation(self.model)
        
        self.results = defaultdict(list)
    
    def run_activation_patching(
        self,
        negated_batch: Dict[str, torch.Tensor],
        non_negated_batch: Dict[str, torch.Tensor],
        layer_idx: int,
        position: Optional[int] = None,
        use_probe: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run activation patching experiment.
        
        Args:
            negated_batch: Batch with negated examples (e.g., "not good")
            non_negated_batch: Batch with non-negated examples (e.g., "good")
            layer_idx: Layer to patch at
            position: Token position to patch (None = all positions)
            use_probe: Whether to use probe for predictions
        
        Returns:
            Dictionary with results including:
            - original_logits: Logits from non-negated batch (baseline)
            - patched_logits: Logits after patching negated activations
            - label_flips: Number of label flips
            - logit_deltas: Change in logits
        """
        # Move batches to device
        negated_batch = {k: v.to(self.device) for k, v in negated_batch.items()}
        non_negated_batch = {k: v.to(self.device) for k, v in non_negated_batch.items()}
        
        # Get original predictions (baseline)
        with torch.no_grad():
            if use_probe and self.probe:
                # Extract hidden states and apply probe
                outputs = self.model(
                    input_ids=non_negated_batch["input_ids"],
                    attention_mask=non_negated_batch["attention_mask"],
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 for embeddings
                # Pool (using CLS for simplicity)
                pooled = hidden_states[:, 0, :]
                original_logits = self.probe(pooled)
            else:
                # Use model's classifier if available
                outputs = self.model(
                    input_ids=non_negated_batch["input_ids"],
                    attention_mask=non_negated_batch["attention_mask"]
                )
                original_logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        # Patch activations from negated to non-negated
        patched_result = self.patcher.patch_activations(
            source_batch=negated_batch,
            target_batch=non_negated_batch,
            layer_idx=layer_idx,
            position=position,
            return_logits=True
        )
        
        # Get patched predictions
        if use_probe and self.probe:
            # Extract patched hidden states and apply probe
            outputs = self.model(
                input_ids=non_negated_batch["input_ids"],
                attention_mask=non_negated_batch["attention_mask"],
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[layer_idx + 1]
            pooled = hidden_states[:, 0, :]
            patched_logits = self.probe(pooled)
        else:
            patched_logits = None
        
        # Compute metrics
        if original_logits is not None and patched_logits is not None:
            original_preds = torch.argmax(original_logits, dim=-1)
            patched_preds = torch.argmax(patched_logits, dim=-1)
            
            label_flips = (original_preds != patched_preds).sum().item()
            logit_deltas = (patched_logits - original_logits).abs().mean().item()
        else:
            label_flips = 0
            logit_deltas = 0.0
        
        results = {
            "original_logits": original_logits,
            "patched_logits": patched_logits,
            "label_flips": label_flips,
            "logit_deltas": logit_deltas,
            "layer_idx": layer_idx,
            "position": position,
        }
        
        self.results["activation_patching"].append(results)
        return results
    
    def run_targeted_ablation(
        self,
        batch: Dict[str, torch.Tensor],
        layer_idx: int,
        dimensions: Union[List[int], torch.Tensor],
        ablation_type: str = "zero",
        position: Optional[int] = None,
        use_probe: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run targeted ablation experiment.
        
        Args:
            batch: Input batch
            layer_idx: Layer to ablate
            dimensions: Dimensions to ablate
            ablation_type: "zero" or "project"
            position: Token position (None = all positions)
            use_probe: Whether to use probe for predictions
        
        Returns:
            Dictionary with ablation results
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions
        with torch.no_grad():
            if use_probe and self.probe:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[layer_idx + 1]
                pooled = hidden_states[:, 0, :]
                original_logits = self.probe(pooled)
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                original_logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        # Perform ablation
        if ablation_type == "zero":
            ablated_hidden = self.ablator.zero_out_dimensions(
                batch, layer_idx, dimensions, position
            )
        elif ablation_type == "project":
            # Create projection matrix (I - P) where P projects onto dimensions
            hidden_size = self.model.config.hidden_size
            projection = torch.eye(hidden_size, device=self.device)
            if isinstance(dimensions, list):
                dimensions = torch.tensor(dimensions, dtype=torch.long)
            projection[dimensions, dimensions] = 0.0
            
            ablated_hidden = self.ablator.project_out_dimensions(
                batch, layer_idx, projection, position
            )
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")
        
        # Get ablated predictions
        if use_probe and self.probe:
            pooled = ablated_hidden[:, 0, :]
            ablated_logits = self.probe(pooled)
        else:
            ablated_logits = None
        
        # Compute metrics
        if original_logits is not None and ablated_logits is not None:
            original_preds = torch.argmax(original_logits, dim=-1)
            ablated_preds = torch.argmax(ablated_logits, dim=-1)
            
            label_flips = (original_preds != ablated_preds).sum().item()
            logit_deltas = (ablated_logits - original_logits).abs().mean().item()
        else:
            label_flips = 0
            logit_deltas = 0.0
        
        results = {
            "original_logits": original_logits,
            "ablated_logits": ablated_logits,
            "label_flips": label_flips,
            "logit_deltas": logit_deltas,
            "layer_idx": layer_idx,
            "dimensions": dimensions,
            "ablation_type": ablation_type,
        }
        
        self.results["targeted_ablation"].append(results)
        return results
    
    def run_control_experiment(
        self,
        batch: Dict[str, torch.Tensor],
        control_type: str = "random",
        layer_idx: Optional[int] = None,
        use_probe: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run control experiments (label-shuffled, representation-shuffled, random).
        
        Args:
            batch: Input batch
            control_type: "label_shuffle", "repr_shuffle", or "random"
            layer_idx: Layer for representation shuffling (if applicable)
            use_probe: Whether to use probe
        
        Returns:
            Dictionary with control results
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions
        with torch.no_grad():
            if use_probe and self.probe:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True
                )
                if layer_idx is not None:
                    hidden_states = outputs.hidden_states[layer_idx + 1]
                else:
                    hidden_states = outputs.hidden_states[-1]
                pooled = hidden_states[:, 0, :]
                original_logits = self.probe(pooled)
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                original_logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        # Apply control
        if control_type == "label_shuffle":
            # Shuffle labels (for probe training control)
            shuffled_labels = batch["labels"][torch.randperm(len(batch["labels"]))]
            # This would be used during probe training, not inference
            control_logits = original_logits  # No change to model
        
        elif control_type == "repr_shuffle":
            # Shuffle representations at a layer
            if layer_idx is None:
                layer_idx = len(outputs.hidden_states) - 2  # Last transformer layer
            
            # Get hidden states and shuffle
            hidden_states = outputs.hidden_states[layer_idx + 1]
            shuffled_indices = torch.randperm(hidden_states.shape[0])
            shuffled_hidden = hidden_states[shuffled_indices]
            
            if use_probe and self.probe:
                pooled = shuffled_hidden[:, 0, :]
                control_logits = self.probe(pooled)
            else:
                control_logits = original_logits
        
        elif control_type == "random":
            # Random intervention (random dimensions zeroed)
            if layer_idx is None:
                layer_idx = len(outputs.hidden_states) - 2
            
            hidden_size = self.model.config.hidden_size
            num_dims = hidden_size // 10  # Ablate 10% randomly
            random_dims = torch.randperm(hidden_size)[:num_dims]
            
            ablated_hidden = self.ablator.zero_out_dimensions(
                batch, layer_idx, random_dims
            )
            
            if use_probe and self.probe:
                pooled = ablated_hidden[:, 0, :]
                control_logits = self.probe(pooled)
            else:
                control_logits = original_logits
        else:
            raise ValueError(f"Unknown control type: {control_type}")
        
        # Compute metrics
        if original_logits is not None and control_logits is not None:
            original_preds = torch.argmax(original_logits, dim=-1)
            control_preds = torch.argmax(control_logits, dim=-1)
            
            label_flips = (original_preds != control_preds).sum().item()
            logit_deltas = (control_logits - original_logits).abs().mean().item()
        else:
            label_flips = 0
            logit_deltas = 0.0
        
        results = {
            "original_logits": original_logits,
            "control_logits": control_logits,
            "label_flips": label_flips,
            "logit_deltas": logit_deltas,
            "control_type": control_type,
            "layer_idx": layer_idx,
        }
        
        self.results["control"].append(results)
        return results
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all interventions."""
        summary = {}
        
        if self.results["activation_patching"]:
            ap_results = self.results["activation_patching"]
            summary["activation_patching"] = {
                "total_experiments": len(ap_results),
                "avg_label_flips": np.mean([r["label_flips"] for r in ap_results]),
                "avg_logit_deltas": np.mean([r["logit_deltas"] for r in ap_results]),
            }
        
        if self.results["targeted_ablation"]:
            ab_results = self.results["targeted_ablation"]
            summary["targeted_ablation"] = {
                "total_experiments": len(ab_results),
                "avg_label_flips": np.mean([r["label_flips"] for r in ab_results]),
                "avg_logit_deltas": np.mean([r["logit_deltas"] for r in ab_results]),
            }
        
        if self.results["control"]:
            ctrl_results = self.results["control"]
            summary["control"] = {
                "total_experiments": len(ctrl_results),
                "avg_label_flips": np.mean([r["label_flips"] for r in ctrl_results]),
                "avg_logit_deltas": np.mean([r["logit_deltas"] for r in ctrl_results]),
            }
        
        return summary

