import json
import math
from pathlib import Path
from collections import defaultdict


QRELS_PATH = Path("data/qrels_sample.jsonl")


def load_qrels(path: Path):
    qrels = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            doc_id = item["doc_id"]
            rel = item["rel"]
            qrels[qid][doc_id] = rel
    return qrels


def dcg_at_k(rels, k):
    dcg = 0.0
    for i in range(min(len(rels), k)):
        dcg += (2 ** rels[i] - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_doc_ids, qrels_for_query, k):
    rels = [qrels_for_query.get(doc_id, 0) for doc_id in ranked_doc_ids]
    dcg = dcg_at_k(rels, k)

    ideal_rels = sorted(qrels_for_query.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def main():
    print("Starte nDCG Evaluation...")

    qrels = load_qrels(QRELS_PATH)

    # Beispiel-Rankings (hier später echte Outputs einlesen)
    run = {
        "Q1": ["D5", "D2", "D3"],
        "Q2": ["D4", "D2", "D3"],
        "Q3": ["D5", "D3", "D2"]
    }

    k = 3
    scores = []

    for qid, ranked_docs in run.items():
        score = ndcg_at_k(ranked_docs, qrels[qid], k)
        scores.append(score)
        print(f"nDCG@{k} für {qid}: {score:.4f}")

    avg = sum(scores) / len(scores)
    print(f"\nDurchschnittliche nDCG@{k}: {avg:.4f}")


if __name__ == "__main__":
    main()
