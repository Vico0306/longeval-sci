import json
from pathlib import Path
from rank_bm25 import BM25Okapi

DOCS_PATH = Path("data/docs_sample.jsonl")
QUERIES_PATH = Path("data/queries_sample.jsonl")


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    # 1) Dokumente laden
    docs = load_jsonl(DOCS_PATH)
    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    # 2) Tokenisierung (sehr simpel für den Anfang)
    tokenized_docs = [text.lower().split() for text in doc_texts]

    # 3) BM25-Modell aufbauen
    bm25 = BM25Okapi(tokenized_docs)

    # 4) Queries laden
    queries = load_jsonl(QUERIES_PATH)

    # 5) Für jede Query: Top-k Dokumente holen
    k = 3
    for q in queries:
        qid = q["id"]
        qtext = q["text"]
        q_tokens = qtext.lower().split()

        scores = bm25.get_scores(q_tokens)
        # Scores mit IDs verbinden
        scored_docs = list(zip(doc_ids, doc_texts, scores))
        # Nach Score sortieren (absteigend)
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n=== Query {qid}: {qtext} ===")
        for rank, (did, dtext, score) in enumerate(scored_docs[:k], start=1):
            print(f"Rank {rank}: {did} (Score={score:.4f})")
            print(f"  {dtext}")


if __name__ == "__main__":
    main()

