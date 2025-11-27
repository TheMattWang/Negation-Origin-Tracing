import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
)


class BaseModule(L.LightningModule):
    """
    Base Lightning Module for sentiment classification with negation.
    Supports two modes:
    1. finetune: Fine-tune the full DistilBERT model
    2. probe: Freeze base model and train linear probes on hidden states
    """
    
    def __init__(
            self, 
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        mode: str = "finetune",
        lr: float = 2e-5,
        probe_layer: Optional[int] = None,
        pooling_strategy: str = "cls",
        probe_lr: float = 1e-3,
            **kwargs
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            mode: "finetune" or "probe"
            lr: Learning rate (for finetune mode)
            probe_layer: Layer index to probe (None = all layers, only for probe mode)
            pooling_strategy: "cls", "mean", or "token" (only for probe mode)
            probe_lr: Learning rate for probe (only for probe mode)
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.mode = mode
        self.lr = lr
        self.probe_layer = probe_layer
        self.pooling_strategy = pooling_strategy
        self.probe_lr = probe_lr
        
        # Load tokenizer for token-based pooling (cache it)
        if pooling_strategy == "token":
            self.tokenizer_cache = AutoTokenizer.from_pretrained(model_name)
            self.not_token_id = self.tokenizer_cache.convert_tokens_to_ids("not")
        else:
            self.tokenizer_cache = None
            self.not_token_id = None
        
        # Load base model
        if mode == "finetune":
            # For fine-tuning, use the classification model
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            # For probing, use base model without classification head
            self.backbone = AutoModel.from_pretrained(model_name)
            # Freeze the base model
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Get hidden size
            hidden_size = self.backbone.config.hidden_size
            
            # Determine which layers to probe
            num_layers = self.backbone.config.num_hidden_layers
            if probe_layer is not None:
                # Probe single layer
                self.probe_layers = [probe_layer]
            else:
                # Probe all layers
                self.probe_layers = list(range(num_layers))
            
            # Create linear probes for each layer
            self.probes = nn.ModuleDict({
                f"layer_{i}": nn.Linear(hidden_size, num_labels)
                for i in self.probe_layers
            })
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', and optionally 'labels'
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        if self.mode == "finetune":
            # Standard fine-tuning
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            result = {"logits": logits}
            if "labels" in batch:
                loss = self.criterion(logits, batch["labels"])
                result["loss"] = loss
            
            return result
        
        else:
            # Probe mode: extract hidden states and apply probes
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get all hidden states (tuple of tensors, one per layer)
            hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
            
            results = {}
            
            # Apply probes to each specified layer
            for layer_idx in self.probe_layers:
                # Get hidden states for this layer
                # hidden_states[0] is embeddings, hidden_states[1] is first transformer layer
                layer_hidden = hidden_states[layer_idx + 1]  # +1 because first is embeddings
                
                # Apply pooling
                pooled = self._pool_hidden_states(layer_hidden, attention_mask, input_ids)
                
                # Apply probe
                probe = self.probes[f"layer_{layer_idx}"]
                logits = probe(pooled)
                
                results[f"logits_layer_{layer_idx}"] = logits
            
            # If probing single layer, return that layer's logits as main output
            if len(self.probe_layers) == 1:
                results["logits"] = results[f"logits_layer_{self.probe_layers[0]}"]
            else:
                # For multiple layers, use the last layer as main output
                results["logits"] = results[f"logits_layer_{self.probe_layers[-1]}"]
            
            # Compute loss if labels provided
            if "labels" in batch:
                # Compute loss for each layer
                losses = {}
                for layer_idx in self.probe_layers:
                    logits = results[f"logits_layer_{layer_idx}"]
                    loss = self.criterion(logits, batch["labels"])
                    losses[f"loss_layer_{layer_idx}"] = loss
                
                # Use average loss across layers, or single layer loss
                if len(losses) == 1:
                    results["loss"] = list(losses.values())[0]
                else:
                    results["loss"] = torch.stack(list(losses.values())).mean()
            
            return results
    
    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states according to the specified strategy.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            input_ids: (batch_size, seq_len)
        
        Returns:
            Pooled representation: (batch_size, hidden_size)
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence (masked)
            # Expand attention mask to match hidden_size
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # Sum hidden states, masking out padding
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            # Count non-padding tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        
        elif self.pooling_strategy == "token":
            # Pool around "not" token
            if self.not_token_id is None:
                # Fallback to [CLS] if tokenizer not initialized
                return hidden_states[:, 0, :]
            
            batch_size = hidden_states.size(0)
            pooled = []
            
            for i in range(batch_size):
                seq = input_ids[i]
                # Find positions where "not" token appears
                not_positions = (seq == self.not_token_id).nonzero(as_tuple=True)[0]
                
                if len(not_positions) > 0:
                    # Use first occurrence of "not"
                    not_idx = not_positions[0].item()
                    # Pool tokens around "not" (not_idx-1 to not_idx+1, or just not_idx)
                    start_idx = max(0, not_idx - 1)
                    end_idx = min(hidden_states.size(1), not_idx + 2)
                    # Only consider non-padding tokens
                    valid_mask = attention_mask[i, start_idx:end_idx].bool()
                    if valid_mask.any():
                        pooled_hidden = hidden_states[i, start_idx:end_idx, :][valid_mask].mean(dim=0)
                    else:
                        pooled_hidden = hidden_states[i, not_idx, :]
                else:
                    # Fallback to [CLS] if "not" not found
                    pooled_hidden = hidden_states[i, 0, :]
                
                pooled.append(pooled_hidden)
            
            return torch.stack(pooled)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(batch)
        loss = outputs["loss"]
        
        # Compute accuracy
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end metrics
        self.train_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self.forward(batch)
        loss = outputs["loss"]
        
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        outputs = self.forward(batch)
        loss = outputs["loss"]
        
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        acc = (preds == labels).float().mean()
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        
        self.test_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self.val_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer based on mode."""
        if self.mode == "finetune":
            # Fine-tune all parameters
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr
            )
        else:
            # Only optimize probe parameters
            optimizer = torch.optim.AdamW(
                self.probes.parameters(),
                lr=self.probe_lr
            )
        
        return optimizer
    
    @classmethod
    def from_args(cls, args):
        """
        Create model from argparse.Namespace.
        
        Args:
            args: argparse.Namespace with model configuration
        
        Returns:
            BaseModule instance
        """
        return cls(
            model_name=getattr(args, "model_name", "distilbert-base-uncased"),
            num_labels=getattr(args, "num_labels", 2),
            mode=getattr(args, "mode", "finetune"),
            lr=getattr(args, "lr", 2e-5),
            probe_layer=getattr(args, "probe_layer", None),
            pooling_strategy=getattr(args, "pooling_strategy", "cls"),
            probe_lr=getattr(args, "probe_lr", 1e-3),
        )
