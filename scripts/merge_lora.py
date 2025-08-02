#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model to create a standalone fine-tuned model.

This script loads the base model and LoRA adapter, merges them, and saves
the resulting model as a complete standalone model that doesn't require
the original base model or PEFT library for inference.

Usage:
    python merge_lora.py --base_model Qwen/Qwen2.5-VL-7B-Instruct \
                        --lora_path /path/to/lora/adapter \
                        --output_path /path/to/merged/model
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True,
        help="Name or path of the base model"
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        required=True,
        help="Path to the LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path where to save the merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for merging (auto, cpu, cuda)"
    )
    return parser.parse_args()


def merge_lora_model(base_model_name: str, lora_path: str, output_path: str, device: str = "auto"):
    """Merge LoRA adapter with base model and save the result."""
    
    print(f"Loading base model: {base_model_name}")
    
    # Determine device mapping
    if device == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    elif device == "cpu":
        device_map = None
    else:
        device_map = {"": device}
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter: {lora_path}")
    
    # Load model with LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA adapter with base model...")
    
    # Merge the adapter with the base model
    merged_model = model.merge_and_unload()
    
    print(f"Loading tokenizer: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    print(f"Saving merged model to: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model and tokenizer
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save model info
    info_file = os.path.join(output_path, "model_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Base model: {base_model_name}\n")
        f.write(f"LoRA adapter: {lora_path}\n")
        f.write(f"Merged by: merge_lora.py\n")
        f.write(f"Device used: {device}\n")
        
        # Calculate model size
        total_params = sum(p.numel() for p in merged_model.parameters())
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Model size (approx): {total_params * 4 / (1024**3):.2f} GB (fp32)\n")
    
    print("✅ Merging completed successfully!")
    print(f"📁 Merged model saved to: {output_path}")
    print(f"📊 Total parameters: {sum(p.numel() for p in merged_model.parameters()):,}")
    
    return merged_model, tokenizer


def main():
    args = parse_args()
    
    # Check if LoRA adapter exists
    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA adapter not found at: {args.lora_path}")
    
    adapter_config = os.path.join(args.lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config}")
    
    try:
        merge_lora_model(args.base_model, args.lora_path, args.output_path, args.device)
    except Exception as e:
        print(f"❌ Error during merging: {e}")
        raise


if __name__ == "__main__":
    main()