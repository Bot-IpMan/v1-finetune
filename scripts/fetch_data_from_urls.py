"""
Utility to fetch textual content from a list of URLs and convert it into
a simple chat training dataset.

The script downloads each page, removes script and style tags, and collapses
whitespace.  Each extracted document is then turned into a single training
example with a generic instruction prompt asking the model to summarise the
page.  The resulting examples are split into training and evaluation sets and
written to separate JSONL files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable, List, Tuple

import requests
from bs4 import BeautifulSoup


def fetch_page_text(url: str, timeout: int = 10) -> str:
    """Download and extract visible text from a web page.

    Args:
        url: The URL to download.
        timeout: How long to wait for the HTTP request.

    Returns:
        The cleaned text content of the page.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def build_examples(texts: Iterable[str]) -> List[dict]:
    """Convert raw page texts into chat training examples.

    For each text we construct a single turn conversation where the user asks
    the assistant to summarise the document.  The assistant answer is left
    empty because during supervised fine‑tuning the model will learn to
    generate the completion.
    """
    examples = []
    for t in texts:
        prompt = f"Будь ласка, зроби короткий стислий виклад наступного тексту: {t}"
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ]
            }
        )
    return examples


def split_data(
    data: List[dict], train_ratio: float = 0.9
) -> Tuple[List[dict], List[dict]]:
    """Shuffle and split data into train and eval sets."""
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]


def write_jsonl(data: List[dict], path: str) -> None:
    """Write a list of dictionaries to a JSON Lines file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def fetch_data_from_urls(
    urls: Iterable[str],
    train_output_path: str,
    eval_output_path: str,
    train_ratio: float = 0.9,
) -> Tuple[List[dict], List[dict]]:
    """Fetch data from the provided URLs and write train/eval JSONL files.

    Returns:
        A tuple of (train_data, eval_data) lists.
    """
    texts = []
    for url in urls:
        try:
            text = fetch_page_text(url)
            if text:
                texts.append(text)
        except Exception as e:
            print(f"Warning: failed to fetch {url}: {e}")
    examples = build_examples(texts)
    train_data, eval_data = split_data(examples, train_ratio=train_ratio)
    write_jsonl(train_data, train_output_path)
    write_jsonl(eval_data, eval_output_path)
    return train_data, eval_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch web pages and build a chat training dataset."
    )
    parser.add_argument(
        "--urls",
        type=str,
        default="",
        help="Comma separated list of URLs to fetch.",
    )
    parser.add_argument(
        "--url_file",
        type=str,
        default="",
        help="Path to a text file containing URLs (one per line).",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/generated/train_urls.jsonl",
        help="Path to write the training JSONL file.",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="data/generated/eval_urls.jsonl",
        help="Path to write the evaluation JSONL file.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of examples to use for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls: List[str] = []
    if args.urls:
        urls += [u.strip() for u in args.urls.split(",") if u.strip()]
    if args.url_file:
        with open(args.url_file, "r", encoding="utf-8") as f:
            urls += [line.strip() for line in f.read().splitlines() if line.strip()]
    if not urls:
        raise ValueError("No URLs provided.")
    fetch_data_from_urls(
        urls,
        args.train_output,
        args.eval_output,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
