from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/paragraphs.jsonl")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir).resolve()
    emb_dir = out_dir / "embeddings"
    faiss_dir = out_dir / "faiss"
    emb_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    texts = [r["text"] for r in rows]

    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(emb_dir / "paragraph_embeddings.npy", embeddings)
    write_jsonl(emb_dir / "paragraph_meta.jsonl", rows)
    faiss.write_index(index, str(faiss_dir / "paragraph.index"))

    print(f"Saved embeddings: {embeddings.shape}")
    print(f"Saved FAISS index to {faiss_dir / 'paragraph.index'}")


if __name__ == "__main__":
    main()