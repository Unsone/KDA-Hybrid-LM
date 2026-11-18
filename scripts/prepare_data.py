"""准备 TinyStories 原版 TXT、训练集 BPE，以及固定的三个 token 划分。"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stories(path):
    parts = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            pieces = line.split("<|endoftext|>")
            for i, piece in enumerate(pieces):
                parts.append(piece)
                if i < len(pieces) - 1:
                    story = "".join(parts).strip()
                    if story:
                        yield story
                    parts = []
        story = "".join(parts).strip()
        if story:
            yield story


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def download(name, directory, revision):
    path = directory / name
    if not path.exists():
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/{name}"
        print(f"Downloading {url}", flush=True)
        temporary = path.with_suffix(".part")
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while True:
                block = source.read(4 * 1024 * 1024)
                if not block:
                    break
                target.write(block)
        temporary.replace(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="Download original TinyStories TXT (~2GB)")
    p.add_argument("--revision", default="f54c09f")
    p.add_argument("--train-text", type=Path)
    p.add_argument("--valid-text", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/tinystories")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--train-tokens", type=int, default=60_000_000)
    p.add_argument("--heldout-tokens", type=int, default=1_000_000)
    p.add_argument("--tokenizer-stories", type=int, default=100_000)
    args = p.parse_args()
    if not 258 <= args.vocab_size <= 65536 or min(args.train_tokens, args.heldout_tokens, args.tokenizer_stories) <= 0:
        p.error("Use vocab-size 258..65536 and positive data budgets")
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError:
        p.error("BPE dependency missing: python -m pip install tokenizers")
    if args.download:
        raw = ROOT / "data/raw/tinystories" / args.revision
        raw.mkdir(parents=True, exist_ok=True)
        args.train_text = download("TinyStories-train.txt", raw, args.revision)
        args.valid_text = download("TinyStories-valid.txt", raw, args.revision)
    if not args.train_text or not args.valid_text:
        p.error("Use --download or supply --train-text and --valid-text in TinyStories TXT format")
    if args.output.exists():
        p.error(f"Output already exists: {args.output}; choose a new --output")
    # 将整个官方验证集排除在训练之外；相同文本始终落入同一 held-out 划分。
    heldout_hashes = {fingerprint(text) for text in stories(args.valid_text)}

    def training_stories(limit=None):
        seen = set()
        for text in stories(args.train_text):
            key = fingerprint(text)
            if key in heldout_hashes or key in seen:
                continue
            seen.add(key)
            yield text
            if limit is not None and len(seen) >= limit:
                break

    print("Training BPE on training stories only...", flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
                                 special_tokens=["[UNK]", "[EOS]"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(training_stories(args.tokenizer_stories), trainer=trainer)
    eos = tokenizer.token_to_id("[EOS]")
    buffers = {name: array("H") for name in ("train", "validation", "test")}
    for text in training_stories():
        ids = tokenizer.encode(text).ids + [eos]
        remaining = args.train_tokens - len(buffers["train"])
        buffers["train"].extend(ids[:remaining])
        if len(buffers["train"]) >= args.train_tokens:
            break
    seen = set()
    for text in stories(args.valid_text):
        key = fingerprint(text)
        if key in seen:
            continue
        seen.add(key)
        split = "test"
        if key[0] % 2 == 0:
            split = "validation"
        remaining = args.heldout_tokens - len(buffers[split])
        if remaining > 0:
            ids = tokenizer.encode(text).ids + [eos]
            buffers[split].extend(ids[:remaining])
    if min(map(len, buffers.values())) < 2:
        raise ValueError("Empty or too-short split; supply more distinct stories")
    args.output.mkdir(parents=True)
    tokenizer.save(str(args.output / "tokenizer.json"))
    metadata = dict(dataset="TinyStories TXT" if args.download else "local TinyStories-format TXT",
                    revision=args.revision if args.download else None,
                    train_text=str(args.train_text.resolve()), valid_text=str(args.valid_text.resolve()),
                    vocab_size=tokenizer.get_vocab_size(), eos=eos, tokenizer_stories=args.tokenizer_stories,
                    train_heldout_exact_dedup=True, token_counts={}, sha256={})
    for name, buffer in buffers.items():
        data = np.frombuffer(buffer, dtype=np.uint16)
        np.save(args.output / f"{name}.npy", data)
        metadata["token_counts"][name] = len(data)
        metadata["sha256"][name] = hashlib.sha256(data.tobytes()).hexdigest()
    metadata["tokenizer_sha256"] = hashlib.sha256((args.output / "tokenizer.json").read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared: {args.output}\nTokens: {metadata['token_counts']}", flush=True)


if __name__ == "__main__":
    main()
