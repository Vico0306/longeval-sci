import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyterrier as pt
import lightgbm as lgb

from pt_dense_features import build_doc_text_cache_for_candidates, add_dense_feature

INDEX_DIR = Path("data/pt_index_v2_month")  # <-- neuer Index
QUERIES_TXT = Path("data/queries.txt")
QRELS_JSONL = Path("data/qrels.jsonl")

RUN_BM25_TEST = Path("runs/pt_bm25_test.jsonl")
RUN_LTR_TEST = Path("runs/pt_ltr_test.jsonl")


def load_queries_txt(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid = parts[0]
            query = " ".join(parts[1:])
            rows.append({"qid": qid, "query": query})
    return pd.DataFrame(rows)


def load_qrels_jsonl(path: Path) -> pd.DataFrame:
    import json
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"qid": str(obj["qid"]), "docno": str(obj["doc_id"]), "rel": int(obj["rel"])})
    return pd.DataFrame(rows)


def fixed_split(qids: List[str], seed: int = 42, test_n: int = 80, val_n: int = 31) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    qids = list(qids)
    rng.shuffle(qids)
    test = qids[:test_n]
    val = qids[test_n:test_n + val_n]
    train = qids[test_n + val_n:]
    return train, val, test


def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col: str = "score") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = {
                "qid": str(r["qid"]),
                "doc_id": str(r["docno"]),
                "rank": int(r["rank"]),
                "score": float(r[score_col]),
            }
            f.write(__import__("json").dumps(obj) + "\n")


def to_int_year(x) -> int:
    try:
        s = str(x)
        return int(s) if s.isdigit() else 0
    except Exception:
        return 0


def to_int_month(x) -> int:
    try:
        s = str(x)
        if s.isdigit():
            v = int(s)
            return v if 1 <= v <= 12 else 0
        return 0
    except Exception:
        return 0


def to_int_yyyymm(x) -> int:
    try:
        s = str(x)
        return int(s) if s.isdigit() and len(s) == 6 else 0
    except Exception:
        return 0


def attach_labels(cands: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    out = cands.merge(qrels, how="left", on=["qid", "docno"])
    out["rel"] = out["rel"].fillna(0).astype(int)
    return out


def candidate_recall_at_k(cands: pd.DataFrame, qrels: pd.DataFrame, k: int) -> float:
    pos = qrels[qrels["rel"] > 0][["qid", "docno"]].drop_duplicates()
    topk = (cands.sort_values(["qid", "rank"])
                .groupby("qid")
                .head(k)[["qid", "docno"]]
                .drop_duplicates())
    merged = pos.merge(topk, on=["qid", "docno"], how="left", indicator=True)
    hit = (merged["_merge"] == "both").sum()
    total = len(pos)
    return float(hit / total) if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--alpha", type=float, default=0.25, help="final_score = bm25 + alpha*ltr")
    ap.add_argument("--recall_k", type=int, default=200, help="Candidate recall@K messen")

    # Time feature toggles (für schnelle Ablations)
    ap.add_argument("--no_year", action="store_true")
    ap.add_argument("--no_month", action="store_true")
    ap.add_argument("--no_yyyymm", action="store_true")

    # Optional: monotone constraint für BM25
    ap.add_argument("--monotone_bm25", action="store_true")

    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    idx = pt.IndexRef.of(str(INDEX_DIR.resolve()))
    queries = load_queries_txt(QUERIES_TXT)
    qrels = load_qrels_jsonl(QRELS_JSONL)

    all_qids = sorted(set(qrels["qid"].astype(str).unique()))
    train_qids, val_qids, test_qids = fixed_split(all_qids, seed=args.seed, test_n=80, val_n=31)
    print(f"[LTR_V6] SPLIT: train={len(train_qids)} | val={len(val_qids)} | test={len(test_qids)}")

    retr = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=args.topk,
        metadata=["docno", "year", "month", "yyyymm", "doclen", "title_len", "abs_len"],
    )

    q_train = queries[queries["qid"].isin(train_qids)].copy()
    q_val = queries[queries["qid"].isin(val_qids)].copy()
    q_test = queries[queries["qid"].isin(test_qids)].copy()

    print("[LTR_V6] BM25 candidates train...")
    cand_train = retr.transform(q_train)
    print("[LTR_V6] BM25 candidates val...")
    cand_val = retr.transform(q_val)
    print("[LTR_V6] BM25 candidates test...")
    cand_test = retr.transform(q_test)

    print(f"[LTR_V6] cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"[LTR_V6] cand_val   rows={len(cand_val)}   | qids={cand_val['qid'].nunique()}")
    print(f"[LTR_V6] cand_test  rows={len(cand_test)}  | qids={cand_test['qid'].nunique()}")

    # Recall sanity
    try:
        rk = min(args.recall_k, args.topk)
        r_train = candidate_recall_at_k(cand_train, qrels, rk)
        r_test = candidate_recall_at_k(cand_test, qrels, rk)
        print(f"[LTR_V6] RECALL@{rk}: train={r_train:.4f} | test={r_test:.4f}")
    except Exception as e:
        print("[LTR_V6] Recall compute failed:", e)

    # ---- DENSE FEATURE ----
    all_cands = pd.concat(
        [cand_train[["qid", "docno"]], cand_val[["qid", "docno"]], cand_test[["qid", "docno"]]],
        ignore_index=True
    )
    build_doc_text_cache_for_candidates(all_cands, verbose=True)

    cand_train = add_dense_feature(cand_train, q_train, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)
    cand_val = add_dense_feature(cand_val, q_val, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)
    cand_test = add_dense_feature(cand_test, q_test, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)

    # ---- FEATURES ----
    for df in (cand_train, cand_val, cand_test):
        df["year_i"] = df["year"].apply(to_int_year).astype(np.int32)
        df["month_i"] = df["month"].apply(to_int_month).astype(np.int32)
        df["yyyymm_i"] = df["yyyymm"].apply(to_int_yyyymm).astype(np.int32)

        df["f_bm25"] = df["score"].astype(np.float32)
        df["f_dense"] = pd.to_numeric(df["f_dense"], errors="coerce").fillna(0).astype(np.float32)

        df["f_qlen"] = df["query"].astype(str).str.len().astype(np.float32)
        df["f_doclen"] = pd.to_numeric(df["doclen"], errors="coerce").fillna(0).astype(np.float32)
        df["f_title_len"] = pd.to_numeric(df["title_len"], errors="coerce").fillna(0).astype(np.float32)
        df["f_abs_len"] = pd.to_numeric(df["abs_len"], errors="coerce").fillna(0).astype(np.float32)

        df["f_year"] = df["year_i"].astype(np.float32)
        df["f_month"] = df["month_i"].astype(np.float32)
        df["f_yyyymm"] = df["yyyymm_i"].astype(np.float32)

    # Feature Auswahl
    feature_cols = ["f_bm25", "f_dense", "f_qlen", "f_doclen", "f_title_len", "f_abs_len"]
    if not args.no_year:
        feature_cols.append("f_year")
    if not args.no_month:
        feature_cols.append("f_month")
    if not args.no_yyyymm:
        feature_cols.append("f_yyyymm")

    print("[LTR_V6] FEATURES:", feature_cols)

    # ---- LABELS ----
    train = attach_labels(cand_train, qrels).sort_values(["qid", "rank"])
    val = attach_labels(cand_val, qrels).sort_values(["qid", "rank"])
    test = attach_labels(cand_test, qrels).sort_values(["qid", "rank"])

    train_group = train.groupby("qid").size().tolist()
    val_group = val.groupby("qid").size().tolist()

    print(f"[LTR_V6] TRAIN: rows={len(train)} | groups={len(train_group)} | positives={(train['rel']>0).sum()}")
    print(f"[LTR_V6] VAL:   rows={len(val)}   | groups={len(val_group)}   | positives={(val['rel']>0).sum()}")

    X_train = train[feature_cols]
    y_train = train["rel"].values
    X_val = val[feature_cols]
    y_val = val["rel"].values

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    if args.monotone_bm25:
        # Constraint-Liste muss exakt feature_cols Reihenfolge matchen
        # nur f_bm25 bekommt 1, der Rest 0
        constraints = [1 if c == "f_bm25" else 0 for c in feature_cols]
        params["monotone_constraints"] = constraints
        print("[LTR_V6] monotone_constraints:", constraints)

    ranker = lgb.LGBMRanker(**params)

    print("[LTR_V6] Training with early stopping...")
    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )
    print("[LTR_V6] MODEL: best_iteration =", ranker.best_iteration_)

    # ---- INFERENCE ----
    test["ltr_score"] = ranker.predict(test[feature_cols], num_iteration=ranker.best_iteration_)

    # Fusion
    test["final_score"] = test["score"].astype(float) + float(args.alpha) * test["ltr_score"].astype(float)
    print(f"[LTR_V6] USING final_score = bm25 + {args.alpha}*ltr")

    # BM25 run
    bm25_out = test.sort_values(["qid", "rank"]).copy()
    bm25_out["rank"] = bm25_out.groupby("qid").cumcount()
    write_run_jsonl(bm25_out[["qid", "docno", "rank", "score"]], RUN_BM25_TEST, score_col="score")
    print("[LTR_V6] WROTE:", RUN_BM25_TEST.resolve())

    # LTR run
    ltr_out = test.sort_values(["qid", "final_score"], ascending=[True, False]).copy()
    ltr_out["rank"] = ltr_out.groupby("qid").cumcount()
    write_run_jsonl(ltr_out[["qid", "docno", "rank", "final_score"]], RUN_LTR_TEST, score_col="final_score")
    print("[LTR_V6] WROTE:", RUN_LTR_TEST.resolve())

    print("[LTR_V6] DONE.")


if __name__ == "__main__":
    main()
