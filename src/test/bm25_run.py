print("BM25 RUN SCRIPT STARTET")

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

DOCS_PATH = Path("data/docs_sample.jsonl")
QUERIES_PATH = Path("data/queries_sample.jsonl")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
RUN_PATH = RUNS_DIR / "bm25_sample.jsonl"


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def main():
    print("Starte BM25-Run-Erstellung...")

    docs = load_jsonl(DOCS_PATH)
    queries = load_jsonl(QUERIES_PATH)

    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    tokenized_docs = [t.lower().split() for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    k = 3
    run_entries = []

    for q in queries:
        qid = q["id"]
        qtext = q["text"]
        print(f"\n=== Query {qid}: {qtext} ===")

        q_tokens = qtext.lower().split()
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_norm = normalize_scores(bm25_scores)

        ranked_indices = np.argsort(-bm25_norm)

        for rank, idx in enumerate(ranked_indices[:k], start=1):
            did = doc_ids[idx]
            score = float(bm25_norm[idx])
            print(f"Rank {rank}: {did} | bm25={score:.4f}")

            run_entries.append({
                "qid": qid,
                "doc_id": did,
                "rank": rank,
                "score": score
            })

    with RUN_PATH.open("w", encoding="utf-8") as f:
        for entry in run_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nBM25-Run-Datei geschrieben nach: {RUN_PATH}")


if __name__ == "__main__":
    main()
