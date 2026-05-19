print("HYBRID SCRIPT WIRD AUSGEFÜHRT")

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sys

DOCS_PATH = Path("data/docs_sample.jsonl")
QUERIES_PATH = Path("data/queries_sample.jsonl")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
RUN_PATH = RUNS_DIR / "hybrid_sample.jsonl"


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def main():
    print("Starte Hybrid-Retrieval...")

    # 1) Dokumente laden
    docs = load_jsonl(DOCS_PATH)
    if not docs:
        print("Keine Dokumente geladen – prüfe data/docs_sample.jsonl")
        return

    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    # 2) BM25 vorbereiten
    tokenized_docs = [t.lower().split() for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    # 3) Dense-Modell laden
    print("Lade SentenceTransformer-Modell...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 4) Dense-Dokument-Embeddings berechnen
    print("Berechne Dokument-Embeddings...")
    doc_emb = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(doc_emb)
    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    # 5) Queries laden
    queries = load_jsonl(QUERIES_PATH)
    if not queries:
        print("Keine Queries geladen – prüfe data/queries_sample.jsonl")
        return

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # Gewichtung BM25 vs Dense
    k = 3        # Top-k Dokumente anzeigen

    run_entries = []  # hier sammeln wir alles für die Run-Datei

    for q in queries:
        qid = q["id"]
        qtext = q["text"]
        print(f"\n=== Query {qid}: {qtext} ===")

        # BM25-Scores
        q_tokens = qtext.lower().split()
        bm25_scores = bm25.get_scores(q_tokens)

        # Dense-Scores
        q_emb = model.encode([qtext], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, len(docs))
        dense_scores_full = np.zeros(len(docs), dtype=np.float32)
        for score, idx in zip(D[0], I[0]):
            dense_scores_full[idx] = score

        bm25_norm = normalize_scores(bm25_scores)
        dense_norm = normalize_scores(dense_scores_full)

        hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_norm

        ranked_indices = np.argsort(-hybrid_scores)

        for rank, idx in enumerate(ranked_indices[:k], start=1):
            did = doc_ids[idx]
            dtext = doc_texts[idx]
            h_score = float(hybrid_scores[idx])
            print(
                f"Rank {rank}: {did} | hybrid={h_score:.4f}, "
                f"bm25={bm25_norm[idx]:.4f}, dense={dense_norm[idx]:.4f}"
            )
            print(f"  {dtext}")

            # Eintrag für Run-File merken
            run_entries.append({
                "qid": qid,
                "doc_id": did,
                "rank": rank,
                "score": h_score
            })

    # Run-Datei schreiben
    with RUN_PATH.open("w", encoding="utf-8") as f:
        for entry in run_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nRun-Datei geschrieben nach: {RUN_PATH}")


if __name__ == "__main__":
    main()
