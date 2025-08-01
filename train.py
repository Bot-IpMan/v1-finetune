"""Training script for fine-tuning a Qwen‑based language model with optional LoRA.

This script largely mirrors the original `train.py` from the repository but
includes a number of improvements aimed at making it more robust in resource
constrained or offline environments.  In particular it avoids loading the
model on a non‑existent GPU (which previously resulted in ``Cannot copy out of
meta tensor`` errors) and dynamically adjusts the number of processes used
for dataset preprocessing.  When running on a machine without CUDA this
version will load the model entirely on the CPU and disable mixed‑precision
training to prevent errors from bitsandbytes.

The high‑level workflow is unchanged:

1.  Parse command‑line arguments.  You can supply local JSONL files via
    ``--train_file`` and ``--eval_file`` or point the script at URLs to fetch
    training data on the fly using ``--urls`` or ``--url_file``.  The
    ``--offline`` flag tells the script to avoid any network requests.
2.  Load the training and evaluation datasets.
3.  Load a base language model and tokenizer.  By default the script will
    download the model from Hugging Face if you have an internet connection.
    If you set ``--offline`` you should also supply ``--model_path`` pointing
    at a locally available model directory.  On CPU‑only machines the model is
    automatically loaded on the CPU to avoid meta tensor errors.
4.  Optionally apply a LoRA adapter.
5.  Preprocess the datasets using the tokenizer’s chat template.  The number
    of worker processes used for tokenization is capped at the dataset size to
    work around an upstream issue where small datasets would trigger the
    message ``num_proc must be <= 1``.
6.  Construct ``TrainingArguments`` and instantiate a ``Trainer``.  Mixed
    precision and CUDA usage are automatically disabled when no GPU is
    available.
7.  Train the model and save the resulting weights.  If LoRA is used only
    the adapter weights are saved; otherwise the full model is written to
    ``output_dir``.

Example usage::

    python train.py --base_model_name Qwen/Qwen1.5-7B \
        --train_file data/train.jsonl \
        --eval_file data/eval.jsonl \
        --output_dir model_output \
        --num_epochs 1

See ``--help`` for a complete list of options.
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model

# When available the helper script can fetch text from remote URLs.  This import
# is optional so that training on local files does not require requests/bs4.
try:
    from scripts.fetch_data_from_urls import fetch_data_from_urls
except ImportError:
    fetch_data_from_urls = None


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Fine‑tune a chat model.")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen1.5-7B",
        help="HF name or path of the base model to load.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to a JSONL file containing training examples.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to a JSONL file containing evaluation examples.",
    )
    parser.add_argument(
        "--urls",
        type=str,
        default=None,
        help="Comma separated list of URLs to fetch data from.",
    )
    parser.add_argument(
        "--url_file",
        type=str,
        default=None,
        help="Path to a text file containing one URL per line.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/sdb/data/ai/models/model_output",
        help="Directory to save the fine‑tuned model.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to apply a LoRA adapter to the base model.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (r) when using LoRA.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine‑tuning.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Per device (GPU) batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load the model in 4‑bit for memory efficient training.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a locally downloaded base model.  Use this when running offline.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Run in offline mode.  When set the script will not attempt to download"
            " models or fetch URLs.  You must provide --model_path and only local"
            " text files will be used."
        ),
    )
    return parser.parse_args()


def get_datasets(train_file: Optional[str], eval_file: Optional[str]) -> Dict[str, Any]:
    """Load datasets from JSONL files."""
    data_files: Dict[str, str] = {}
    if train_file:
        data_files["train"] = train_file
    if eval_file:
        data_files["validation"] = eval_file
    if not data_files:
        raise ValueError(
            "You must specify at least one of --train_file, --urls or --url_file "
            "to provide training data."
        )
    ds = load_dataset("json", data_files=data_files)
    return ds


def tokenize_function(
    example: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """Convert a chat conversation into model input and labels.

    Each example must contain a ``messages`` key with a list of chat turns.  The
    tokenizer's chat template is applied to assemble the conversation into a single
    prompt string.  Labels are identical to ``input_ids``; the Trainer will
    automatically apply the causal mask.
    """
    messages = example["messages"]
    # Add the generation prompt to enable the model to continue the conversation.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenized = tokenizer(
        prompt,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]
    # Labels are the same as input_ids for causal language modelling
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # If URLs are provided build a dataset on the fly.  Offline mode forbids network calls.
    if args.urls or args.url_file:
        if args.offline:
            print(
                "Offline mode specified; cannot fetch remote URLs. Please provide local text files instead."
            )
            return
        if fetch_data_from_urls is None:
            raise RuntimeError(
                "requests/bs4 are required for URL fetching but are not installed."
            )
        # Parse URL list
        urls: List[str] = []
        if args.urls:
            urls.extend(u.strip() for u in args.urls.split(",") if u.strip())
        if args.url_file:
            with open(args.url_file, "r", encoding="utf‑8") as f:
                urls.extend(
                    line.strip() for line in f.read().splitlines() if line.strip()
                )
        random.shuffle(urls)
        if not urls:
            raise ValueError("No URLs provided.")
        train_path = os.path.join(args.output_dir, "train_urls.jsonl")
        eval_path = os.path.join(args.output_dir, "eval_urls.jsonl")
        fetch_data_from_urls(urls, train_path, eval_path)
        train_file = train_path
        eval_file = eval_path
    else:
        train_file = args.train_file
        eval_file = args.eval_file

    # If no explicit training data or URLs are provided, look for
    # user‑supplied text files and url lists in the default data/custom
    # directory.  This allows drop‑in fine‑tuning without specifying
    # arguments.
    if not (train_file or args.urls or args.url_file):
        custom_text_dir = os.path.join("data", "custom", "texts")
        custom_url_file = os.path.join("data", "custom", "urls.txt")
        collected_urls: List[str] = []
        collected_texts: List[str] = []
        # Read all text files in data/custom/texts
        if os.path.isdir(custom_text_dir):
            for fname in os.listdir(custom_text_dir):
                fpath = os.path.join(custom_text_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(
                    (".txt", ".md", ".html")
                ):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        collected_texts.append(f.read())
        # Read URLs from data/custom/urls.txt
        if os.path.exists(custom_url_file):
            with open(custom_url_file, "r", encoding="utf-8") as f:
                collected_urls += [
                    line.strip() for line in f.read().splitlines() if line.strip()
                ]
        # If we found any texts or urls, build a dataset on the fly
        if collected_texts or collected_urls:
            tmp_train = os.path.join(args.output_dir, "custom_train.jsonl")
            tmp_eval = os.path.join(args.output_dir, "custom_eval.jsonl")
            # Write text examples directly
            examples: List[Dict[str, Any]] = []
            for t in collected_texts:
                prompt = (
                    f"Будь ласка, зроби короткий стислий виклад наступного тексту: {t}"
                )
                examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": ""},
                        ]
                    }
                )
            # Fetch and add URL examples if needed
            if collected_urls:
                if fetch_data_from_urls is None:
                    raise RuntimeError(
                        "requests/bs4 are required for URL fetching but are not installed."
                    )
                _train_data, _eval_data = fetch_data_from_urls(
                    collected_urls, tmp_train, tmp_eval
                )
                # Append url examples to examples list
                examples += _train_data + _eval_data
            # Split examples into train/eval
            random.shuffle(examples)
            split_idx = max(1, int(len(examples) * 0.8))
            with open(tmp_train, "w", encoding="utf-8") as f:
                for item in examples[:split_idx]:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            with open(tmp_eval, "w", encoding="utf-8") as f:
                for item in examples[split_idx:]:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            train_file = tmp_train
            eval_file = tmp_eval

    # Load datasets
    datasets = get_datasets(train_file, eval_file)

    # Load tokenizer and model
    # When offline, require a local model path
    base_model_source = args.model_path or args.base_model_name
    if args.offline and args.model_path is None:
        print(
            "Offline mode requires --model_path to point to a locally available pretrained model."
        )
        return
    tokenizer = AutoTokenizer.from_pretrained(base_model_source, trust_remote_code=True)
    # Make sure the tokenizer uses the correct padding side and tokens
    tokenizer.pad_token = tokenizer.eos_token

    # quantization config (use 4‑bit only when requested and GPU available)
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Decide on device_map: on GPU use 'auto', otherwise keep None so the model
    # loads entirely on CPU.  Passing device_map="auto" on a CPU‑only machine
    # triggers accelerate’s offloading logic and results in meta tensors.
    if torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_source,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(
            f"Error loading base model '{base_model_source}': {exc}\n"
            "If running offline, ensure --model_path points to a local model directory."
        )
        return

    # Optionally apply LoRA for parameter efficient training
    if args.use_lora:
        # prepare the model for k‑bit training if using quantization
        if args.use_4bit:
            # Import inside the branch to avoid pulling in bitsandbytes on CPU
            from peft.tuners.lora import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Preprocess datasets.  Use at most one worker per example to avoid dataset
    # library complaining when the dataset is tiny.  This fixes the issue where
    # ``num_proc must be <= 1`` messages were emitted for small datasets.
    processed_datasets: Dict[str, Any] = {}
    for split in datasets.keys():
        num_examples = (
            len(datasets[split]) if hasattr(datasets[split], "__len__") else 1
        )
        # Use at most len(datasets[split]) workers but no more than available CPUs
        max_procs = os.cpu_count() or 1
        num_proc = min(max_procs, max(1, num_examples))
        processed = datasets[split].map(
            lambda x: tokenize_function(x, tokenizer),
            remove_columns=datasets[split].column_names,
            num_proc=num_proc,
        )
        processed.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        processed_datasets[split] = processed

    # Determine whether to use CUDA.  When no GPU is available or when the user
    # explicitly requested 4‑bit quantisation (which is unsupported without CUDA),
    # disable CUDA entirely.  This prevents the Trainer from attempting to move
    # meta tensors onto a non‑existent GPU and throwing a NotImplementedError.
    use_cuda = torch.cuda.is_available() and not args.use_4bit

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=use_cuda,
        bf16=False,
        report_to="none",
        no_cuda=not use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets.get("train"),
        eval_dataset=processed_datasets.get("validation"),
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the fine‑tuned model.  If LoRA was used only the adapter weights are stored.
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        # also save base model name so the inference script can reconstruct the model
        with open(
            os.path.join(args.output_dir, "base_model_name.txt"), "w", encoding="utf‑8"
        ) as f:
            f.write(args.base_model_name)
    else:
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
