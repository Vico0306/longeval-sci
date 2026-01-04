import json
from pathlib import Path
import random

import pandas as pd
import pyterrier as pt

# ---- Paths / Settings ----
INDEX_DIR = Path("data/pt_index")
QUERIES_TXT = Path("data/queries.txt")
QRELS_JSONL = Path("data/qrels.jsonl")
QRELS_TREC_TXT = Path("data/qrels.txt")

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 1000
SEED = 42

TRAIN_FRAC = 0.72
VAL_FRAC = 0.08  # rest is test (0.20)

EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 300

# LightGBM LambdaMART-ish defaults
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
}


# ---- IO helpers ----
def read_queries_tsv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: qid <tab> query
            if "\t" in line:
                qid, query = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                qid, query = parts
            rows.append({"qid": qid.strip(), "query": query.strip()})
    return pd.DataFrame(rows)


def read_qrels_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append({"qid": str(o["qid"]), "docno": str(o["doc_id"]), "label": int(o["rel"])})
    return pd.DataFrame(rows)


def read_qid_to_snapshot(path: Path) -> dict:
    """
    qrels.txt format you showed:
      qid snapshot docid rel
      e.g. 2eb8... 2024-11 41260840 2
    We only need first snapshot per qid (same for all lines of that qid).
    Returns: {qid: (year:int, month:int)}
    """
    m = {}
    if not path.exists():
        return m

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            qid = parts[0]
            snap = parts[1]
            if qid in m:
                continue

            # expected "YYYY-MM"
            try:
                if "-" in snap:
                    y, mo = snap.split("-", 1)
                    m[qid] = (int(y), int(mo))
                else:
                    m[qid] = (int(snap[:4]), 1)
            except Exception:
                # fallback unknown
                m[qid] = (0, 0)
    return m


# ---- Feature Engineering ----
def add_features(df: pd.DataFrame, qid_to_snap: dict) -> pd.DataFrame:
    """
    Input df expected columns: qid, docno, score, query, doclen, title_len, abs_len, year
    Output adds:
      f_bm25, f_qlen, f_doclen, f_title_len, f_abs_len, f_year
      f_snap_year, f_snap_month
      f_age_months, f_is_recent_24m
    """
    out = df.copy()

    # BM25 score
    out["f_bm25"] = out["score"].astype(float)

    # query length (token count)
    out["f_qlen"] = out["query"].fillna("").apply(lambda s: len(str(s).split())).astype(int)

    # document / field lengths from Terrier meta
    # (these are strings sometimes, cast carefully)
    for col in ["doclen", "title_len", "abs_len"]:
        if col in out.columns:
            out[f"f_{col}"] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float)
        else:
            out[f"f_{col}"] = 0.0

    # year meta: sometimes empty string -> NaN -> 0
    if "year" in out.columns:
        out["f_year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    else:
        out["f_year"] = 0

    # snapshot year/month from qrels.txt
    def snap_y(qid):
        return qid_to_snap.get(qid, (0, 0))[0]

    def snap_m(qid):
        return qid_to_snap.get(qid, (0, 0))[1]

    out["f_snap_year"] = out["qid"].apply(snap_y).astype(int)
    out["f_snap_month"] = out["qid"].apply(snap_m).astype(int)

    # age in months: snapshot - doc_date
    # if doc_year missing -> age=0 (neutral)
    def age_months(row):
        sy = int(row["f_snap_year"])
        sm = int(row["f_snap_month"])
        dy = int(row["f_year"])
        if sy <= 0 or sm <= 0 or dy <= 0:
            return 0
        snap = sy * 12 + sm
        doc = dy * 12 + 1
        return max(0, snap - doc)

    out["f_age_months"] = out.apply(age_months, axis=1).astype(int)
    out["f_is_recent_24m"] = (out["f_age_months"] <= 24).astype(int)

    return out


# ---- Train/Val/Test split on qids ----
def split_qids(qids, seed=SEED):
    qids = list(qids)
    random.Random(seed).shuffle(qids)
    n = len(qids)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = set(qids[:n_train])
    val = set(qids[n_train:n_train + n_val])
    test = set(qids[n_train + n_val:])
    return train, val, test


# ---- Run writing ----
def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col="score"):
    """
    Writes JSONL in your expected run format:
      {"qid": "...", "doc_id": "...", "rank": i, "score": ...}
    Assumes df has columns qid, docno, rank, score_col
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "qid": str(r["qid"]),
                "doc_id": str(r["docno"]),
                "rank": int(r["rank"]),
                "score": float(r[score_col]),
            }) + "\n")


def main():
    # init Java
    if not pt.java.started():
        pt.java.init()

    # load inputs
    queries = read_queries_tsv(QUERIES_TXT)
    qrels = read_qrels_jsonl(QRELS_JSONL)
    qid_to_snap = read_qid_to_snapshot(QRELS_TREC_TXT)

    all_qids = sorted(set(qrels["qid"].unique()))
    train_qids, val_qids, test_qids = split_qids(all_qids, seed=SEED)
    print(f"SPLIT: train={len(train_qids)} | val={len(val_qids)} | test={len(test_qids)}")

    # load index as absolute IndexRef (robust on Windows)
    idx_ref = pt.IndexRef.of(str(INDEX_DIR.resolve()))

    # BM25 candidate retriever
    bm25 = pt.terrier.Retriever(
        idx_ref,
        wmodel="BM25",
        num_results=TOPK,
        metadata=["docno", "year", "doclen", "title_len", "abs_len"],
    )

    # candidates for each split
    def get_candidates(qdf: pd.DataFrame, tag: str) -> pd.DataFrame:
        print(f"BM25: candidates {tag}...")
        cand = bm25.transform(qdf[["qid", "query"]])
        # keep only necessary cols
        # cand has: qid, docid, docno, year, doclen, title_len, abs_len, rank, score, query
        return cand

    q_train = queries[queries["qid"].isin(train_qids)].copy()
    q_val = queries[queries["qid"].isin(val_qids)].copy()
    q_test = queries[queries["qid"].isin(test_qids)].copy()

    cand_train = get_candidates(q_train, "train")
    cand_val = get_candidates(q_val, "val")
    cand_test = get_candidates(q_test, "test")

    print(f"cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"cand_val   rows={len(cand_val)}   | qids={cand_val['qid'].nunique()}")
    print(f"cand_test  rows={len(cand_test)}  | qids={cand_test['qid'].nunique()}")

    # add labels to candidates
    qrels_small = qrels[["qid", "docno", "label"]].copy()
    train_l = cand_train.merge(qrels_small, on=["qid", "docno"], how="left").fillna({"label": 0})
    val_l = cand_val.merge(qrels_small, on=["qid", "docno"], how="left").fillna({"label": 0})
    test_l = cand_test.merge(qrels_small, on=["qid", "docno"], how="left").fillna({"label": 0})

    # feature engineering (includes snapshot age!)
    train_l = add_features(train_l, qid_to_snap)
    val_l = add_features(val_l, qid_to_snap)
    test_l = add_features(test_l, qid_to_snap)

    feature_cols = [
        "f_bm25", "f_qlen", "f_doclen", "f_title_len", "f_abs_len",
        "f_year", "f_snap_year", "f_snap_month", "f_age_months", "f_is_recent_24m"
    ]
    print("FEATURES:", feature_cols)

    # group sizes per qid (required by LGBM ranking)
    def make_groups(df):
        grp = df.groupby("qid").size().tolist()
        return grp

    # drop any qids with 0 candidates (rare, but can happen)
    # (lightgbm will error if groups mismatch)
    def filter_nonempty(df):
        keep = df.groupby("qid").size()
        keep_qids = set(keep[keep > 0].index)
        return df[df["qid"].isin(keep_qids)].copy()

    train_l = filter_nonempty(train_l)
    val_l = filter_nonempty(val_l)
    test_l = filter_nonempty(test_l)

    print(f"TRAIN: rows={len(train_l)} | groups={train_l['qid'].nunique()}")
    print(f"VAL:   rows={len(val_l)}   | groups={val_l['qid'].nunique()}")

    # build matrices
    import lightgbm as lgb

    X_train = train_l[feature_cols]
    y_train = train_l["label"].astype(int)
    g_train = make_groups(train_l)

    X_val = val_l[feature_cols]
    y_val = val_l["label"].astype(int)
    g_val = make_groups(val_l)

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train)
    dval = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain)

    print(f"Training until validation scores don't improve for {EARLY_STOPPING_ROUNDS} rounds")

    model = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        num_boost_round=MAX_TREES,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)]
    )

    best_iter = model.best_iteration or MAX_TREES
    print(f"MODEL: trained. best_iteration = {best_iter}")

    # scoring on test candidates
    test_l = test_l.copy()
    test_l["ltr_score"] = model.predict(test_l[feature_cols], num_iteration=best_iter)

    # write BM25 test run
    bm25_run = test_l.sort_values(["qid", "score"], ascending=[True, False]).copy()
    bm25_run["rank"] = bm25_run.groupby("qid").cumcount()
    bm25_out = RUNS_DIR / "pt_bm25_test.jsonl"
    write_run_jsonl(bm25_run[["qid", "docno", "rank", "score"]], bm25_out, score_col="score")
    print("WROTE:", bm25_out.resolve())

    # write LTR test run
    ltr_run = test_l.sort_values(["qid", "ltr_score"], ascending=[True, False]).copy()
    ltr_run["rank"] = ltr_run.groupby("qid").cumcount()
    ltr_out = RUNS_DIR / "pt_ltr_test.jsonl"
    write_run_jsonl(ltr_run[["qid", "docno", "rank", "ltr_score"]], ltr_out, score_col="ltr_score")
    print("WROTE:", ltr_out.resolve())

    print("DONE.")


if __name__ == "__main__":
    main()
