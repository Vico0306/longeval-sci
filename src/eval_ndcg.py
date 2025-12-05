import json
import math
import sys
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


def load_run(path: Path):
    run = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            doc_id = item["doc_id"]
            rank = item.get("rank")
            score = item.get("score", 0.0)
            run[qid].append((doc_id, rank, score))

    # pro Query nach Rank sortieren (falls vorhanden), sonst nach Score
    run_sorted = {}
    for qid, entries in run.items():
        if entries and entries[0][1] is not None:
            entries.sort(key=lambda x: x[1])  # nach Rank
        else:
            entries.sort(key=lambda x: x[2], reverse=True)  # nach Score
        run_sorted[qid] = [doc_id for doc_id, _, _ in entries]
    return run_sorted


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
    if len(sys.argv) < 2:
        print("Usage: python src/eval_ndcg.py runs/hybrid_sample.jsonl")
        return

    run_path = Path(sys.argv[1])
    if not run_path.exists():
        print(f"Run-Datei nicht gefunden: {run_path}")
        return

    print(f"Starte nDCG Evaluation für Run: {run_path}")

    qrels = load_qrels(QRELS_PATH)
    run = load_run(run_path)

    k = 3
    scores = []

    for qid, ranked_docs in run.items():
        if qid not in qrels:
            # keine Relevanzinfos für diese Query → überspringen
            continue
        score = ndcg_at_k(ranked_docs, qrels[qid], k)
        scores.append(score)
        print(f"nDCG@{k} für {qid}: {score:.4f}")

    if not scores:
        print("Keine überlappenden Queries zwischen Run und Qrels gefunden.")
        return

    avg = sum(scores) / len(scores)
    print(f"\nDurchschnittliche nDCG@{k}: {avg:.4f}")


if __name__ == "__main__":
    main()
