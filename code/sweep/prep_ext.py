"""Tokenize additional shards (002, 003) into {edu,web}_train_ext.bin for high-M runs.
The original train files are untouched, so the main grid's data order is preserved; the loader
appends the extension after the original permutation (train_sweep.Loader(ext=True))."""
import json, os, sys
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "processed", "sweep")
tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json")); eot = tok.token_to_id("<eot>")
SRC = {"edu": "fineweb_edu", "web": "fineweb"}
meta = json.load(open(os.path.join(OUT, "meta.json")))
for reg, folder in SRC.items():
    n_tok = n_bytes = 0
    with open(os.path.join(OUT, f"{reg}_train_ext.bin"), "wb") as f:
        for shard in ["002_00000.parquet", "003_00000.parquet"]:
            pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", folder, shard))
            for b in pf.iter_batches(batch_size=1000, columns=["text"]):
                texts = b.column(0).to_pylist()
                for enc, t in zip(tok.encode_batch(texts), texts):
                    ids = enc.ids + [eot]
                    f.write(np.asarray(ids, dtype=np.uint16).tobytes()); n_tok += len(ids); n_bytes += len(t.encode())
    meta[reg]["train_ext"] = {"tokens": n_tok, "bytes": n_bytes, "shards": "002,003"}
    print(reg, n_tok, flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
