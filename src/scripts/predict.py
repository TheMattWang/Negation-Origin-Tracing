import argparse
import torch
from transformers import AutoTokenizer

from src.models import BaseModule


def main():
    """Run inference on a single text or file of texts."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to classify.",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="File with one text per line to classify.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save predictions (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on.",
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = BaseModule.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to(args.device)
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    
    # Get texts to predict
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.text_file:
        with open(args.text_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide either --text or --text_file")
    
    print(f"\nClassifying {len(texts)} text(s)...\n")
    
    # Run predictions
    predictions = []
    label_map = {0: "negative", 1: "positive"}  # SST-2 labels
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            
            # Move to device
            input_ids = encoding["input_ids"].to(args.device)
            attention_mask = encoding["attention_mask"].to(args.device)
            
            # Predict
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            outputs = model(batch)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred].item()
            
            label = label_map[pred]
            predictions.append({
                "text": text,
                "label": label,
                "confidence": confidence,
                "probs": probs[0].cpu().tolist(),
            })
            
            print(f"Text: {text}")
            print(f"  Prediction: {label} (confidence: {confidence:.3f})")
            print(f"  Probabilities: negative={probs[0][0]:.3f}, positive={probs[0][1]:.3f}")
            print()
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            for pred in predictions:
                f.write(f"{pred['text']}\t{pred['label']}\t{pred['confidence']:.3f}\n")
        print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()

