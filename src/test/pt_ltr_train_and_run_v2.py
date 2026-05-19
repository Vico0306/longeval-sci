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
TRAIN_TOPK = 500   # etwas mehr als vorher
TEST_FRAC = 0.2
SEED = 42

MODEL_NUM_LEAVES = 63
MODEL_NUM_TREES = 400
MODEL_LEARNING_RATE = 0.05


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
                "doc_id": str(row.docno),   # docno ist eure echte CORE id
                "rank": int(row.new_rank),
                "score": float(getattr(row, score_col)),
            }) + "\n")


def main():
    ensure_java()

    index_path = INDEX_DIR.resolve()
    idx = pt.IndexRef.of(str(index_path))

    queries = read_queries_tsv(QUERIES_PATH.resolve())
    qrels = read_qrels_jsonl(QRELS_PATH.resolve())

    # split by qid
    qids = queries["qid"].tolist()
    qids_shuffled = pd.Series(qids).sample(frac=1.0, random_state=SEED).tolist()
    cut = int(len(qids_shuffled) * (1 - TEST_FRAC))
    train_qids = set(qids_shuffled[:cut])
    test_qids = set(qids_shuffled[cut:])

    q_train = queries[queries["qid"].isin(train_qids)].reset_index(drop=True)
    q_test  = queries[queries["qid"].isin(test_qids)].reset_index(drop=True)

    print(f"SPLIT: train={len(q_train)} | test={len(q_test)}")

    # Retriever mit Meta!
    bm25 = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=CAND_TOPK,
        metadata=["docno", "year", "doclen", "title_len", "abs_len"]
    )

    print("BM25: candidates train...")
    cand_train = bm25.transform(q_train)
    print("BM25: candidates test...")
    cand_test = bm25.transform(q_test)

    print(f"cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"cand_test rows={len(cand_test)} | qids={cand_test['qid'].nunique()}")

    # Features
    qlen_map_train = dict(zip(q_train["qid"], q_train["query"].map(qlen)))
    qlen_map_test  = dict(zip(q_test["qid"],  q_test["query"].map(qlen)))

    def prep(df: pd.DataFrame, qlen_map: dict) -> pd.DataFrame:
        out = df.copy()
        out["f_bm25"] = out["score"].astype(float)
        out["f_qlen"] = out["qid"].map(qlen_map).fillna(0).astype(int)

        # meta kommt als string -> sicher casten
        for col in ["doclen", "title_len", "abs_len", "year"]:
            out[col] = out[col].fillna("0").astype(str)

        out["f_doclen"] = pd.to_numeric(out["doclen"], errors="coerce").fillna(0).astype(int)
        out["f_title_len"] = pd.to_numeric(out["title_len"], errors="coerce").fillna(0).astype(int)
        out["f_abs_len"] = pd.to_numeric(out["abs_len"], errors="coerce").fillna(0).astype(int)
        out["f_year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)

        return out

    cand_train = prep(cand_train, qlen_map_train)
    cand_test  = prep(cand_test, qlen_map_test)

    # Train labels join
    train_df = cand_train.merge(qrels, on=["qid", "docno"], how="left")
    train_df["label"] = train_df["label"].fillna(0).astype(int)

    # Limit per query
    train_df = train_df.sort_values(["qid", "rank"]).groupby("qid").head(TRAIN_TOPK).reset_index(drop=True)

    feature_cols = ["f_bm25", "f_qlen", "f_doclen", "f_title_len", "f_abs_len", "f_year"]
    X = train_df[feature_cols]
    y = train_df["label"]
    group_sizes = train_df.groupby("qid").size().tolist()

    print(f"TRAIN: rows={len(train_df)} | groups={len(group_sizes)} | features={feature_cols}")

    import lightgbm as lgb
    lgb_train = lgb.Dataset(X, label=y, group=group_sizes, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": MODEL_LEARNING_RATE,
        "num_leaves": MODEL_NUM_LEAVES,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": SEED,
    }

    model = lgb.train(params=params, train_set=lgb_train, num_boost_round=MODEL_NUM_TREES)
    print("MODEL: trained.")

    # ---- write BM25 test run ----
    bm25_test = cand_test.sort_values(["qid", "rank"]).reset_index(drop=True)
    bm25_test["new_rank"] = bm25_test.groupby("qid").cumcount()
    bm25_test["bm25_score"] = bm25_test["score"].astype(float)
    write_run_jsonl(bm25_test, RUN_BM25_TEST, "bm25_score")
    print(f"WROTE: {RUN_BM25_TEST.resolve()}")

    # ---- rerank test with LTR ----
    ltr_test = cand_test.copy()
    ltr_test["ltr_score"] = model.predict(ltr_test[feature_cols])
    ltr_test = ltr_test.sort_values(["qid", "ltr_score"], ascending=[True, False])
    ltr_test["new_rank"] = ltr_test.groupby("qid").cumcount()
    write_run_jsonl(ltr_test, RUN_LTR_TEST, "ltr_score")
    print(f"WROTE: {RUN_LTR_TEST.resolve()}")

    print("DONE.")


if __name__ == "__main__":
    main()
