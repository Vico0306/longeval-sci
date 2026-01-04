import json
import random
from pathlib import Path

import pandas as pd
import pyterrier as pt


# -------------------------
# Paths / Settings
# -------------------------
INDEX_DIR = Path("data/pt_index")
QUERIES_TXT = Path("data/queries.txt")
QRELS_JSONL = Path("data/qrels.jsonl")
QRELS_TREC_TXT = Path("data/qrels.txt")
SPLIT_JSON = Path("splits/split_seed42.json")

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate pool for retrieval (bigger pool helps)
TOPK_CAND = 1000

# For run output (keep 1000, official runs usually need deep ranking)
TOPK_RUN = 1000

# Training sampling
MAX_UNJUDGED_NEG_PER_Q = 200   # sample from unjudged docs per query
SEED = 42

EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 800

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
}


# -------------------------
# IO helpers
# -------------------------
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
            try:
                if "-" in snap:
                    y, mo = snap.split("-", 1)
                    m[qid] = (int(y), int(mo))
                else:
                    m[qid] = (int(snap[:4]), 1)
            except Exception:
                m[qid] = (0, 0)
    return m


def load_fixed_split(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Split-Datei fehlt: {path} (make_split.py ausführen)")
    split = json.loads(path.read_text(encoding="utf-8"))
    return set(split["train"]), set(split["val"]), set(split["test"])


# -------------------------
# Feature Engineering
# -------------------------
def add_features(df: pd.DataFrame, qid_to_snap: dict) -> pd.DataFrame:
    out = df.copy()

    out["f_bm25"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).astype(float)
    out["f_qlen"] = out["query"].fillna("").apply(lambda s: len(str(s).split())).astype(int)

    for col in ["doclen", "title_len", "abs_len"]:
        out[f"f_{col}"] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0).astype(float)

    out["f_year"] = pd.to_numeric(out.get("year", 0), errors="coerce").fillna(0).astype(int)

    out["f_snap_year"] = out["qid"].apply(lambda q: qid_to_snap.get(q, (0, 0))[0]).astype(int)
    out["f_snap_month"] = out["qid"].apply(lambda q: qid_to_snap.get(q, (0, 0))[1]).astype(int)

    def age_months(row):
        sy = int(row["f_snap_year"])
        sm = int(row["f_snap_month"])
        dy = int(row["f_year"])
        if sy <= 0 or sm <= 0 or dy <= 0:
            return 0
        snap = sy * 12 + sm
        doc = dy * 12 + 1
        return max(0, snap - doc)

    out["f_age_months"] = out.apply(age_months, axis=1)
    out["f_age_months"] = pd.to_numeric(out["f_age_months"], errors="coerce").fillna(0).clip(0, 240).astype(int)
    out["f_age_years"] = (out["f_age_months"] / 12.0).astype(float)
    out["f_is_recent_24m"] = (out["f_age_months"] <= 24).astype(int)

    return out


# -------------------------
# Training sampling (key improvement!)
# -------------------------
def sample_training_rows(cand: pd.DataFrame, qrels: pd.DataFrame, max_unjudged_neg: int, seed: int) -> pd.DataFrame:
    """
    cand: candidates (qid, docno, score, query, ...)
    qrels: judged pairs (qid, docno, label) where label can be 0..2
    We keep:
      - all positives (label > 0)
      - all judged-0 negatives (label == 0)
      - sample up to max_unjudged_neg from unjudged docs per query
    Unjudged sampled get label=0
    """
    rnd = random.Random(seed)

    merged = cand.merge(qrels, on=["qid", "docno"], how="left")
    # label: NaN => unjudged
    # We'll mark them and sample later
    merged["is_unjudged"] = merged["label"].isna()
    merged["label"] = merged["label"].fillna(-1).astype(int)

    rows = []
    for qid, g in merged.groupby("qid", sort=False):
        pos = g[g["label"] > 0]
        judged0 = g[g["label"] == 0]
        unjudged = g[g["label"] == -1]

        # sample unjudged
        if len(unjudged) > max_unjudged_neg:
            # stable-ish sampling
            idxs = list(unjudged.index)
            rnd.shuffle(idxs)
            unjudged = unjudged.loc[idxs[:max_unjudged_neg]]

        # convert sampled unjudged to label 0
        if len(unjudged) > 0:
            unjudged = unjudged.copy()
            unjudged["label"] = 0

        out = pd.concat([pos, judged0, unjudged], axis=0, ignore_index=True)
        rows.append(out)

    outdf = pd.concat(rows, axis=0, ignore_index=True)
    return outdf


# -------------------------
# Run writing
# -------------------------
def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col="score", topk=1000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["qid", score_col], ascending=[True, False]).copy()
    df["rank"] = df.groupby("qid").cumcount()
    df = df[df["rank"] < topk]

    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "qid": str(r["qid"]),
                "doc_id": str(r["docno"]),
                "rank": int(r["rank"]),
                "score": float(r[score_col]),
            }) + "\n")


def make_groups(df: pd.DataFrame):
    return df.groupby("qid").size().tolist()


# -------------------------
# Main
# -------------------------
def main():
    if not pt.java.started():
        pt.java.init()

    train_qids, val_qids, test_qids = load_fixed_split(SPLIT_JSON)
    print(f"SPLIT(FIXED): train={len(train_qids)} | val={len(val_qids)} | test={len(test_qids)}")

    queries = read_queries_tsv(QUERIES_TXT)
    qrels = read_qrels_jsonl(QRELS_JSONL)
    qid_to_snap = read_qid_to_snapshot(QRELS_TREC_TXT)

    q_train = queries[queries["qid"].isin(train_qids)].copy()
    q_val = queries[queries["qid"].isin(val_qids)].copy()
    q_test = queries[queries["qid"].isin(test_qids)].copy()

    idx_ref = pt.IndexRef.of(str(INDEX_DIR.resolve()))

    bm25 = pt.terrier.Retriever(
        idx_ref,
        wmodel="BM25",
        num_results=TOPK_CAND,
        metadata=["docno", "year", "doclen", "title_len", "abs_len"],
    )

    def get_candidates(qdf: pd.DataFrame, tag: str) -> pd.DataFrame:
        print(f"BM25: candidates {tag}...")
        return bm25.transform(qdf[["qid", "query"]])

    cand_train = get_candidates(q_train, "train")
    cand_val = get_candidates(q_val, "val")
    cand_test = get_candidates(q_test, "test")

    print(f"cand_train rows={len(cand_train)} | qids={cand_train['qid'].nunique()}")
    print(f"cand_val   rows={len(cand_val)}   | qids={cand_val['qid'].nunique()}")
    print(f"cand_test  rows={len(cand_test)}  | qids={cand_test['qid'].nunique()}")

    # --- KEY: sample training rows to reduce noisy negatives ---
    train_l = sample_training_rows(cand_train, qrels, MAX_UNJUDGED_NEG_PER_Q, SEED)
    val_l = sample_training_rows(cand_val, qrels, MAX_UNJUDGED_NEG_PER_Q, SEED)

    # test: keep all candidates, labels only for analysis (not required)
    test_l = cand_test.merge(qrels, on=["qid", "docno"], how="left").fillna({"label": 0})

    train_l = add_features(train_l, qid_to_snap)
    val_l = add_features(val_l, qid_to_snap)
    test_l = add_features(test_l, qid_to_snap)

    feature_cols = [
        "f_bm25", "f_qlen", "f_doclen", "f_title_len", "f_abs_len",
        "f_year", "f_snap_year", "f_snap_month", "f_age_years", "f_is_recent_24m"
    ]
    print("FEATURES:", feature_cols)

    # sanity stats
    print("TRAIN positives:", int((train_l["label"] > 0).sum()), "of", len(train_l))
    print("VAL positives:", int((val_l["label"] > 0).sum()), "of", len(val_l))

    # build LGB datasets
    import lightgbm as lgb

    dtrain = lgb.Dataset(
        train_l[feature_cols],
        label=train_l["label"].astype(int),
        group=make_groups(train_l),
    )
    dval = lgb.Dataset(
        val_l[feature_cols],
        label=val_l["label"].astype(int),
        group=make_groups(val_l),
        reference=dtrain,
    )

    print(f"Training until validation scores don't improve for {EARLY_STOPPING_ROUNDS} rounds")
    model = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        num_boost_round=MAX_TREES,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)],
    )

    best_iter = model.best_iteration or MAX_TREES
    print(f"MODEL: trained. best_iteration = {best_iter}")

    # predict test
    test_l = test_l.copy()
    test_l["ltr_score"] = model.predict(test_l[feature_cols], num_iteration=best_iter)

    # write runs (BM25 + LTR)
    bm25_out = RUNS_DIR / "pt_bm25_test.jsonl"
    ltr_out = RUNS_DIR / "pt_ltr_test.jsonl"

    write_run_jsonl(test_l, bm25_out, score_col="score", topk=TOPK_RUN)
    write_run_jsonl(test_l, ltr_out, score_col="ltr_score", topk=TOPK_RUN)

    print("WROTE:", bm25_out.resolve())
    print("WROTE:", ltr_out.resolve())
    print("DONE.")


if __name__ == "__main__":
    main()
