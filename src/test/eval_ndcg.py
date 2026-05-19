import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def dcg(rels):
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(run_docids, qrels_dict, k):
    # run_docids: list of doc_id in rank order
    rels = [qrels_dict.get(doc_id, 0) for doc_id in run_docids[:k]]
    dcg_val = dcg(rels)

    ideal_rels = sorted(qrels_dict.values(), reverse=True)[:k]
    idcg_val = dcg(ideal_rels)

    if idcg_val == 0:
        return None  # keine relevanten docs für diese query
    return dcg_val / idcg_val


def load_qrels_jsonl(path: Path):
    qrels = defaultdict(dict)  # qid -> {doc_id: rel}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["qid"])
            doc_id = str(obj["doc_id"])
            rel = int(obj["rel"])
            qrels[qid][doc_id] = rel
    return qrels


def load_run_jsonl(path: Path, k):
    # Wir speichern nur Top-k pro Query
    run = defaultdict(list)  # qid -> [doc_id,...] in rank order
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["qid"])
            doc_id = str(obj.get("doc_id") or obj.get("docno") or obj.get("docid"))
            if doc_id is None:
                continue
            # rank kann fehlen; wir nehmen Dateireihenfolge als rank
            if len(run[qid]) < k:
                run[qid].append(str(doc_id))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_path", type=str, help="Pfad zur Run-JSONL Datei")
    ap.add_argument("--qrels", type=str, default="data/qrels.jsonl", help="Pfad zur Qrels-JSONL Datei")
    ap.add_argument("--k", type=int, default=10, help="nDCG@k")
    args = ap.parse_args()

    run_path = Path(args.run_path)
    qrels_path = Path(args.qrels)
    k = args.k

    print(f"Starte nDCG Evaluation für Run: {run_path}")
    print(f"Qrels: {qrels_path} | k={k}")

    qrels = load_qrels_jsonl(qrels_path)
    run = load_run_jsonl(run_path, k)

    qrels_qids = set(qrels.keys())
    run_qids = set(run.keys())
    overlap = qrels_qids & run_qids

    print(f"Qrels queries: {len(qrels_qids)} | Run queries: {len(run_qids)} | Overlap: {len(overlap)}")

    if not overlap:
        print("Keine überlappenden Queries zwischen Run und Qrels gefunden.")
        return

    scores = []
    skipped_no_rels = 0

    for qid in sorted(overlap):
        s = ndcg_at_k(run[qid], qrels[qid], k)
        if s is None:
            skipped_no_rels += 1
            continue
        scores.append(s)

    if not scores:
        print("Keine Queries mit relevanten Dokumenten in Qrels (idcg=0).")
        return

    avg = sum(scores) / len(scores)
    print(f"Queries mit Score: {len(scores)} | Übersprungen (keine rel docs): {skipped_no_rels}")
    print(f"Durchschnittliche nDCG@{k}: {avg:.4f}")


if __name__ == "__main__":
    main()
