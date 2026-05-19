import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyterrier as pt
import lightgbm as lgb

from pt_dense_features import build_doc_text_cache_for_candidates, add_dense_feature


INDEX_DIR = Path("data/pt_index")
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
            # format: qid <tab or spaces> query
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
    """
    Deterministic split by shuffling with seed.
    """
    rng = np.random.default_rng(seed)
    qids = list(qids)
    rng.shuffle(qids)
    test = qids[:test_n]
    val = qids[test_n : test_n + val_n]
    train = qids[test_n + val_n :]
    return train, val, test


def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col: str = "score") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # expects columns: qid, docno, rank, score_col
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = {
                "qid": str(r["qid"]),
                "doc_id": str(r["docno"]),
                "rank": int(r["rank"]),
                "score": float(r[score_col]),
            }
            f.write(__import__("json").dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    idx = pt.IndexRef.of(str(INDEX_DIR.resolve()))
    queries = load_queries_txt(QUERIES_TXT)
    qrels = load_qrels_jsonl(QRELS_JSONL)

    all_qids = sorted(set(qrels["qid"].astype(str).unique()))
    train_qids, val_qids, test_qids = fixed_split(all_qids, seed=args.seed, test_n=80, val_n=31)

    print(f"SPLIT(FIXED): train={len(train_qids)} | val={len(val_qids)} | test={len(test_qids)}")

    # Retriever gives us BM25 score + useful metadata
    retr = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=args.topk,
        metadata=["docno", "year", "doclen", "title_len", "abs_len"],
    )

    q_train = queries[queries["qid"].isin(train_qids)].copy()
    q_val = queries[queries["qid"].isin(val_qids)].copy()
    q_test = queries[queries["qid"].isin(test_qids)].copy()

    print("BM25: candidates train...")
    cand_train = retr.transform(q_train)
    print("BM25: candidates val...")
    cand_val = retr.transform(q_val)
    print("BM25: candidates test...")
    cand_test = retr.transform(q_test)

    print(f"cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"cand_val   rows={len(cand_val)} | qids={cand_val['qid'].nunique()}")
    print(f"cand_test  rows={len(cand_test)} | qids={cand_test['qid'].nunique()}")

    # ---- DENSE FEATURE ----
    # Build cache once for all candidates (train+val+test) to avoid rescans later
    all_cands = pd.concat([cand_train[["qid", "docno"]], cand_val[["qid", "docno"]], cand_test[["qid", "docno"]]], ignore_index=True)
    build_doc_text_cache_for_candidates(all_cands, verbose=True)

    cand_train = add_dense_feature(cand_train, q_train, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)
    cand_val = add_dense_feature(cand_val, q_val, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)
    cand_test = add_dense_feature(cand_test, q_test, model_name=args.dense_model, batch_size=args.batch_size, verbose=True)

    # ---- FEATURE ENGINEERING ----
    # Normalize/clean year
    def to_int_year(x):
        try:
            s = str(x)
            return int(s) if s.isdigit() else 0
        except Exception:
            return 0

    for df in (cand_train, cand_val, cand_test):
        df["year_i"] = df["year"].apply(to_int_year).astype(np.int32)
        df["f_bm25"] = df["score"].astype(np.float32)
        df["f_qlen"] = df["query"].astype(str).str.len().astype(np.float32)
        df["f_doclen"] = pd.to_numeric(df["doclen"], errors="coerce").fillna(0).astype(np.float32)
        df["f_title_len"] = pd.to_numeric(df["title_len"], errors="coerce").fillna(0).astype(np.float32)
        df["f_abs_len"] = pd.to_numeric(df["abs_len"], errors="coerce").fillna(0).astype(np.float32)
        df["f_year"] = df["year_i"].astype(np.float32)
        # f_dense already exists

    feature_cols = ["f_bm25", "f_dense", "f_qlen", "f_doclen", "f_title_len", "f_abs_len", "f_year"]
    print("FEATURES:", feature_cols)

    # ---- LABELS ----
    # Join qrels onto candidates; rel>0 as positive (you can also use graded rel)
    def attach_labels(cands: pd.DataFrame) -> pd.DataFrame:
        out = cands.merge(qrels, how="left", on=["qid", "docno"])
        out["rel"] = out["rel"].fillna(0).astype(int)
        return out

    train = attach_labels(cand_train)
    val = attach_labels(cand_val)
    test = attach_labels(cand_test)

    # Group sizes for LightGBM ranking
    train = train.sort_values(["qid", "rank"])
    val = val.sort_values(["qid", "rank"])
    test = test.sort_values(["qid", "rank"])

    train_group = train.groupby("qid").size().tolist()
    val_group = val.groupby("qid").size().tolist()

    print(f"TRAIN: rows={len(train)} | groups={len(train_group)} | positives={(train['rel']>0).sum()}")
    print(f"VAL:   rows={len(val)}   | groups={len(val_group)}   | positives={(val['rel']>0).sum()}")

    X_train = train[feature_cols].values
    y_train = train["rel"].values
    X_val = val[feature_cols].values
    y_val = val["rel"].values

    # ---- MODEL ----
    ranker = lgb.LGBMRanker(
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

    print("Training with early stopping...")
    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    print("MODEL: best_iteration =", ranker.best_iteration_)

    # ---- INFERENCE (test rerank) ----
    test["ltr_score"] = ranker.predict(test[feature_cols].values, num_iteration=ranker.best_iteration_)

    # Build BM25 test run
    bm25_out = test.copy()
    bm25_out = bm25_out.sort_values(["qid", "rank"])
    bm25_out["rank"] = bm25_out.groupby("qid").cumcount()
    write_run_jsonl(bm25_out[["qid", "docno", "rank", "score"]], RUN_BM25_TEST, score_col="score")
    print("WROTE:", RUN_BM25_TEST.resolve())

    # Build LTR reranked run
    ltr_out = test.copy()
    ltr_out = ltr_out.sort_values(["qid", "ltr_score"], ascending=[True, False])
    ltr_out["rank"] = ltr_out.groupby("qid").cumcount()
    write_run_jsonl(ltr_out[["qid", "docno", "rank", "ltr_score"]], RUN_LTR_TEST, score_col="ltr_score")
    print("WROTE:", RUN_LTR_TEST.resolve())

    print("DONE.")


if __name__ == "__main__":
    main()
