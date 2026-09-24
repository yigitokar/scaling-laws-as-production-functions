"""Prepare training corpora for the local scaling sweeps.

Two data regimes ("labs" with different data quality):
  edu : FineWeb-Edu sample-10BT (educational-quality filtered web text)
  web : FineWeb sample-10BT (generic filtered web text)

One shared byte-level BPE tokenizer (vocab 8192) is trained on an equal mix of both,
so losses are measured in the same units for every lab (common output measure).
Validation sets are held out from each corpus; every model is evaluated on both.

Outputs (data/processed/sweep/):
  tokenizer.json
  {edu,web}_train.bin, {edu,web}_val.bin   (uint16 token ids, documents separated by <eot>)
  meta.json                                 (token counts, bytes per token for bits-per-byte)
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed", "sweep")
os.makedirs(OUT, exist_ok=True)
VOCAB = 8192
VAL_DOCS = 20000
TOKENIZER_TRAIN_CHARS = 300_000_000  # per corpus
SOURCES = {"edu": os.path.join(RAW, "fineweb_edu"), "web": os.path.join(RAW, "fineweb")}


def iter_texts(folder, batch_size=2000):
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".parquet"):
            continue
        pf = pq.ParquetFile(os.path.join(folder, fn))
        for b in pf.iter_batches(batch_size=batch_size, columns=["text"]):
            yield from b.column(0).to_pylist()


def tokenizer_corpus():
    for name, folder in SOURCES.items():
        n = 0
        for i, t in enumerate(iter_texts(folder)):
            if i < VAL_DOCS:  # never train the tokenizer on validation docs
                continue
            yield t
            n += len(t)
            if n > TOKENIZER_TRAIN_CHARS:
                break


def train_tokenizer():
    path = os.path.join(OUT, "tokenizer.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<eot>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(tokenizer_corpus(), trainer=trainer)
    tok.save(path)
    return tok


def encode_corpus(tok, name, folder, max_train_tokens=2_500_000_000):
    eot = tok.token_to_id("<eot>")
    meta = {}
    for split in ["val", "train"]:
        out_path = os.path.join(OUT, f"{name}_{split}.bin")
        buf, n_tok, n_bytes, batch = [], 0, 0, []
        with open(out_path, "wb") as f:
            for i, t in enumerate(iter_texts(folder)):
                if (split == "val") != (i < VAL_DOCS):
                    if split == "val":
                        break
                    continue
                batch.append(t)
                if len(batch) == 1000:
                    for enc, txt in zip(tok.encode_batch(batch), batch):
                        ids = enc.ids + [eot]
                        f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                        n_tok += len(ids)
                        n_bytes += len(txt.encode("utf-8"))
                    batch = []
                    if n_tok > max_train_tokens:
                        break
            if batch:
                for enc, txt in zip(tok.encode_batch(batch), batch):
                    ids = enc.ids + [eot]
                    f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    n_tok += len(ids)
                    n_bytes += len(txt.encode("utf-8"))
        meta[split] = {"tokens": int(n_tok), "bytes": int(n_bytes), "bytes_per_token": n_bytes / max(n_tok, 1)}
        print(name, split, meta[split], flush=True)
    return meta


if __name__ == "__main__":
    tok = train_tokenizer()
    print("tokenizer vocab", tok.get_vocab_size(), flush=True)
    meta = {"vocab_size": tok.get_vocab_size()}
    for name, folder in SOURCES.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        meta[name] = encode_corpus(tok, name, folder)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
