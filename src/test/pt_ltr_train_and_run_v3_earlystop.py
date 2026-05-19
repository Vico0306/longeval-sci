from pathlib import Path
import json
import re
import pandas as pd
import pyterrier as pt

INDEX_DIR = Path("data/pt_index")
QUERIES_PATH = Path("data/queries.txt")
QRELS_PATH = Path("data/qrels.jsonl")

RUN_BM25_TEST = Path("runs/pt_bm25_test.jsonl")
RUN_LTR_TEST  = Path("runs/pt_ltr_test.jsonl")

CAND_TOPK = 1000
TRAIN_TOPK = 500

TEST_FRAC = 0.2
VAL_FRAC = 0.1     # Anteil aus trainval
SEED = 42

MODEL_NUM_LEAVES = 31
MODEL_NUM_TREES = 2000          # ruhig hoch, early stopping stoppt eh
MODEL_LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 50


def read_queries_tsv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                qid, query = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"Bad query line: {line}")
                qid, query = parts
            rows.append({"qid": str(qid).strip(), "query": str(query).strip()})
    return pd.DataFrame(rows)


def read_qrels_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"qid": str(obj["qid"]), "docno": str(obj["doc_id"]), "label": int(obj["rel"])})
    return pd.DataFrame(rows)


def qlen(text: str) -> int:
    return len(re.findall(r"\w+", text.lower()))


def ensure_java():
    if not pt.java.started():
        pt.java.init()


def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write(json.dumps({
                "qid": str(row.qid),
                "doc_id": str(row.docno),
                "rank": int(row.new_rank),
                "score": float(getattr(row, score_col)),
            }) + "\n")


def prep_features(df: pd.DataFrame, qlen_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["f_bm25"] = out["score"].astype(float)
    out["f_qlen"] = out["qid"].map(qlen_map).fillna(0).astype(int)

    for col in ["doclen", "title_len", "abs_len", "year"]:
        out[col] = out[col].fillna("0").astype(str)

    out["f_doclen"] = pd.to_numeric(out["doclen"], errors="coerce").fillna(0).astype(int)
    out["f_title_len"] = pd.to_numeric(out["title_len"], errors="coerce").fillna(0).astype(int)
    out["f_abs_len"] = pd.to_numeric(out["abs_len"], errors="coerce").fillna(0).astype(int)
    out["f_year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    return out


def head_per_qid(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sort_values(["qid", "rank"]).groupby("qid").head(n).reset_index(drop=True)


def main():
    ensure_java()

    idx = pt.IndexRef.of(str(INDEX_DIR.resolve()))
    queries = read_queries_tsv(QUERIES_PATH.resolve())
    qrels = read_qrels_jsonl(QRELS_PATH.resolve())

    # --- split train/val/test ---
    qids = queries["qid"].tolist()
    qids_shuf = pd.Series(qids).sample(frac=1.0, random_state=SEED).tolist()

    cut_test = int(len(qids_shuf) * (1 - TEST_FRAC))
    trainval = qids_shuf[:cut_test]
    test_qids = set(qids_shuf[cut_test:])

    cut_val = int(len(trainval) * (1 - VAL_FRAC))
    train_qids = set(trainval[:cut_val])
    val_qids = set(trainval[cut_val:])

    q_train = queries[queries["qid"].isin(train_qids)].reset_index(drop=True)
    q_val   = queries[queries["qid"].isin(val_qids)].reset_index(drop=True)
    q_test  = queries[queries["qid"].isin(test_qids)].reset_index(drop=True)

    print(f"SPLIT: train={len(q_train)} | val={len(q_val)} | test={len(q_test)}")

    bm25 = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=CAND_TOPK,
        metadata=["docno", "year", "doclen", "title_len", "abs_len"]
    )

    print("BM25: candidates train...")
    cand_train = bm25.transform(q_train)
    print("BM25: candidates val...")
    cand_val = bm25.transform(q_val)
    print("BM25: candidates test...")
    cand_test = bm25.transform(q_test)

    print(f"cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"cand_val   rows={len(cand_val)}   | qids={cand_val['qid'].nunique()}")
    print(f"cand_test  rows={len(cand_test)}  | qids={cand_test['qid'].nunique()}")

    # feature prep
    qlen_map_train = dict(zip(q_train["qid"], q_train["query"].map(qlen)))
    qlen_map_val   = dict(zip(q_val["qid"],   q_val["query"].map(qlen)))
    qlen_map_test  = dict(zip(q_test["qid"],  q_test["query"].map(qlen)))

    cand_train = prep_features(cand_train, qlen_map_train)
    cand_val   = prep_features(cand_val,   qlen_map_val)
    cand_test  = prep_features(cand_test,  qlen_map_test)

    # label join (train/val)
    train_df = cand_train.merge(qrels, on=["qid", "docno"], how="left")
    train_df["label"] = train_df["label"].fillna(0).astype(int)
    train_df = head_per_qid(train_df, TRAIN_TOPK)

    val_df = cand_val.merge(qrels, on=["qid", "docno"], how="left")
    val_df["label"] = val_df["label"].fillna(0).astype(int)
    val_df = head_per_qid(val_df, TRAIN_TOPK)

    feature_cols = ["f_bm25", "f_qlen", "f_doclen", "f_title_len", "f_abs_len", "f_year"]

    import lightgbm as lgb
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    g_train = train_df.groupby("qid").size().tolist()

    X_val = val_df[feature_cols]
    y_val = val_df["label"]
    g_val = val_df.groupby("qid").size().tolist()

    print(f"TRAIN: rows={len(train_df)} | groups={len(g_train)}")
    print(f"VAL:   rows={len(val_df)}   | groups={len(g_val)}")
    print("FEATURES:", feature_cols)

    lgb_train = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=False)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   group=g_val,   free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": MODEL_LEARNING_RATE,
        "num_leaves": MODEL_NUM_LEAVES,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": SEED,
    }

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=MODEL_NUM_TREES,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)],
    )
    print("MODEL: trained. best_iteration =", model.best_iteration)

    # --- write BM25 test run ---
    bm25_test = cand_test.sort_values(["qid", "rank"]).reset_index(drop=True)
    bm25_test["new_rank"] = bm25_test.groupby("qid").cumcount()
    bm25_test["bm25_score"] = bm25_test["score"].astype(float)
    write_run_jsonl(bm25_test, RUN_BM25_TEST, "bm25_score")
    print(f"WROTE: {RUN_BM25_TEST.resolve()}")

    # --- rerank test with LTR ---
    ltr_test = cand_test.copy()
    ltr_test["ltr_score"] = model.predict(ltr_test[feature_cols], num_iteration=model.best_iteration)
    ltr_test = ltr_test.sort_values(["qid", "ltr_score"], ascending=[True, False])
    ltr_test["new_rank"] = ltr_test.groupby("qid").cumcount()
    write_run_jsonl(ltr_test, RUN_LTR_TEST, "ltr_score")
    print(f"WROTE: {RUN_LTR_TEST.resolve()}")
    print("DONE.")


if __name__ == "__main__":
    main()
