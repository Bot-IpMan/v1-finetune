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
5.  Preprocess the datasets using the tokenizer's chat template.  The number
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

    python train.py --base_model_name Qwen/Qwen2.5-Coder-7B \
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
import logging
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        default="Qwen/Qwen2.5-Coder-7B",
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
        help="Per device (GPU/CPU) batch size.",
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
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to YAML configuration file to load default parameters.",
    )
    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed. Cannot load config file.")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")
        return {}


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.use_4bit and not torch.cuda.is_available():
        logger.warning("4-bit quantization requires CUDA but no GPU available. Disabling 4-bit.")
        args.use_4bit = False
    
    if args.offline and not args.model_path:
        raise ValueError("Offline mode requires --model_path to point to a locally available model.")
    
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive integers.")
    
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")


def get_datasets(train_file: Optional[str], eval_file: Optional[str]) -> Dict[str, Any]:
    """Load datasets from JSONL files."""
    data_files: Dict[str, str] = {}
    if train_file and os.path.exists(train_file):
        data_files["train"] = train_file
    if eval_file and os.path.exists(eval_file):
        data_files["validation"] = eval_file
    
    if not data_files:
        raise ValueError(
            "You must specify at least one of --train_file, --urls or --url_file "
            "to provide training data."
        )
    
    logger.info(f"Loading datasets from: {list(data_files.keys())}")
    ds = load_dataset("json", data_files=data_files)
    return ds


def tokenize_function(
    example: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int = 2048
) -> Dict[str, Any]:
    """Tokenize a dataset example for causal language modelling.

    The function is flexible with respect to input format:

    - If ``messages`` is present, it should be a list of chat turns compatible
      with the tokenizer's chat template.
    - If ``prompt``/``completion`` fields are present, they will be converted
      into a two turn conversation of user → assistant.
    - If a plain ``text`` field is provided, the text is used directly for
      language‑model style training without any chat template.

    Regardless of the input style, labels mirror ``input_ids`` so that the
    trainer performs standard causal language modelling.
    """

    if "messages" in example:
        messages = example["messages"]
        add_gen_prompt = messages[-1].get("role") != "assistant"
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen_prompt
        )
    elif "prompt" in example and "completion" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    elif "text" in example:
        prompt = example["text"]
    else:
        raise KeyError(
            "Dataset examples must contain 'messages', 'prompt'/'completion' or 'text' keys"
        )

    tokenized = tokenizer(
        prompt,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,  # Don't pad during tokenization, Trainer will handle this
    )
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]
    labels = input_ids.clone()  # Labels are the same as input_ids for causal LM
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collect_custom_data(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    """Collect custom training data from default directories."""
    custom_text_dir = os.path.join("data", "custom", "texts")
    custom_url_file = os.path.join("data", "custom", "urls.txt")
    collected_urls: List[str] = []
    collected_texts: List[str] = []
    
    # Read all text files in data/custom/texts
    if os.path.isdir(custom_text_dir):
        for fname in os.listdir(custom_text_dir):
            fpath = os.path.join(custom_text_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".md", ".html")):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty texts
                            collected_texts.append(text)
                            logger.info(f"Loaded text file: {fname}")
                except Exception as e:
                    logger.warning(f"Failed to read {fpath}: {e}")
    
    # Read URLs from data/custom/urls.txt
    if os.path.exists(custom_url_file) and not args.offline:
        try:
            with open(custom_url_file, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f.read().splitlines() if line.strip()]
                collected_urls.extend(urls)
                logger.info(f"Loaded {len(urls)} URLs from {custom_url_file}")
        except Exception as e:
            logger.warning(f"Failed to read {custom_url_file}: {e}")
    
    # If we found any texts or urls, build a dataset on the fly
    if not (collected_texts or collected_urls):
        return None, None
    
    tmp_train = os.path.join(args.output_dir, "custom_train.jsonl")
    tmp_eval = os.path.join(args.output_dir, "custom_eval.jsonl")

    def _summarize(text: str) -> Optional[str]:
        """Generate a brief summary for ``text``.

        Attempts to use ``sumy`` when available; otherwise returns ``None`` to
        signal that the text should be skipped."""
        try:
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.summarizers.lsa import LsaSummarizer

            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            sentences = summarizer(parser.document, 1)
            summary = " ".join(str(s) for s in sentences).strip()
            return summary or None
        except Exception as e:  # pragma: no cover - best effort
            logger.warning(f"Failed to summarize text: {e}")
            return None

    # Write text examples directly
    examples: List[Dict[str, Any]] = []
    for t in collected_texts:
        summary = _summarize(t)
        if not summary:
            logger.warning("Skipping text without a generated summary")
            continue
        prompt = (
            "Будь ласка, зроби короткий стислий виклад наступного тексту: "
            f"{t}"
        )
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": summary},
                ]
            }
        )
    
    # Fetch and add URL examples if needed
    if collected_urls:
        if fetch_data_from_urls is None:
            raise RuntimeError(
                "requests/bs4 are required for URL fetching but are not installed."
            )
        logger.info(f"Fetching data from {len(collected_urls)} URLs...")
        _train_data, _eval_data = fetch_data_from_urls(
            collected_urls, tmp_train, tmp_eval
        )
        # Append url examples to examples list
        examples += _train_data + _eval_data
    
    # Split examples into train/eval
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * 0.8))
    
    os.makedirs(os.path.dirname(tmp_train), exist_ok=True)
    with open(tmp_train, "w", encoding="utf-8") as f:
        for item in examples[:split_idx]:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    with open(tmp_eval, "w", encoding="utf-8") as f:
        for item in examples[split_idx:]:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"Created custom datasets: {len(examples[:split_idx])} train, {len(examples[split_idx:])} eval examples")
    return tmp_train, tmp_eval


def setup_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    """Load and setup model and tokenizer."""
    base_model_source = args.model_path or args.base_model_name
    
    logger.info(f"Loading tokenizer from {base_model_source}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup quantization config
    quantization_config = None
    if args.use_4bit and torch.cuda.is_available():
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Setup device mapping
    device_map = "auto" if torch.cuda.is_available() else None
    
    logger.info(f"Loading model from {base_model_source} (device_map={device_map})")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_source,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as exc:
        logger.error(f"Error loading base model '{base_model_source}': {exc}")
        if args.offline:
            logger.error("If running offline, ensure --model_path points to a local model directory.")
        raise
    
    return model, tokenizer


def setup_lora(model, args: argparse.Namespace):
    """Setup LoRA if requested."""
    if not args.use_lora:
        return model
    
    logger.info("Setting up LoRA adapter")
    
    # Prepare the model for k‑bit training if using quantization
    if args.use_4bit and torch.cuda.is_available():
        from peft.tuners.lora import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def process_datasets(datasets: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Process datasets with tokenization."""
    processed_datasets: Dict[str, Any] = {}
    
    for split in datasets.keys():
        logger.info(f"Processing {split} dataset...")
        
        num_examples = len(datasets[split]) if hasattr(datasets[split], "__len__") else 1
        # Use at most len(datasets[split]) workers but no more than available CPUs
        max_procs = os.cpu_count() or 1
        num_proc = min(max_procs, max(1, num_examples))
        
        processed = datasets[split].map(
            lambda x: tokenize_function(x, tokenizer),
            remove_columns=datasets[split].column_names,
            num_proc=num_proc,
            desc=f"Tokenizing {split}",
        )
        processed.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        processed_datasets[split] = processed
        logger.info(f"Processed {len(processed)} examples for {split}")
    
    return processed_datasets


def main() -> None:
    args = parse_args()
    
    # Load config file if provided
    if args.config_file:
        config = load_config_file(args.config_file)
        # Update args with config values (args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Training arguments: {vars(args)}")
    
    if args.offline:
        # Disable all network calls from Hugging Face libraries
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Running in offline mode")

    # Determine training data sources
    train_file, eval_file = args.train_file, args.eval_file
    
    # Handle URL-based data fetching
    if args.urls or args.url_file:
        if args.offline:
            logger.error("Offline mode specified; cannot fetch remote URLs. Please provide local text files instead.")
            return
        
        if fetch_data_from_urls is None:
            raise RuntimeError("requests/bs4 are required for URL fetching but are not installed.")
        
        # Parse URL list
        urls: List[str] = []
        if args.urls:
            urls.extend(u.strip() for u in args.urls.split(",") if u.strip())
        if args.url_file:
            with open(args.url_file, "r", encoding="utf-8") as f:
                urls.extend(line.strip() for line in f.read().splitlines() if line.strip())
        
        if not urls:
            raise ValueError("No URLs provided.")
        
        random.shuffle(urls)
        train_path = os.path.join(args.output_dir, "train_urls.jsonl")
        eval_path = os.path.join(args.output_dir, "eval_urls.jsonl")
        logger.info(f"Fetching data from {len(urls)} URLs...")
        fetch_data_from_urls(urls, train_path, eval_path)
        train_file, eval_file = train_path, eval_path
    
    # Handle custom data directory
    elif not train_file:
        train_file, eval_file = collect_custom_data(args)
    
    # Load datasets
    datasets = get_datasets(train_file, eval_file)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Setup LoRA if requested
    model = setup_lora(model, args)
    
    # Process datasets
    processed_datasets = process_datasets(datasets, tokenizer)
    
    # Setup training arguments with corrected logic
    has_cuda = torch.cuda.is_available()
    use_mixed_precision = has_cuda and not args.use_4bit  # Only use mixed precision on GPU without 4bit
    
    logger.info(f"Training setup - CUDA available: {has_cuda}, Mixed precision: {use_mixed_precision}")
    
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
        eval_strategy="epoch" if "validation" in processed_datasets else "no",
        save_strategy="epoch",
        fp16=use_mixed_precision,  # Only use fp16 on GPU without 4bit
        bf16=False,
        report_to="none",
        no_cuda=not has_cuda,
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets.get("train"),
        eval_dataset=processed_datasets.get("validation"),
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the fine‑tuned model
    logger.info(f"Saving model to {args.output_dir}")
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        # Save base model name for inference script
        with open(os.path.join(args.output_dir, "base_model_name.txt"), "w", encoding="utf-8") as f:
            f.write(args.base_model_name)
        logger.info("Saved LoRA adapter weights")
    else:
        trainer.save_model(args.output_dir)
        logger.info("Saved full model")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()