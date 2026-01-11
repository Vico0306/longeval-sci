import argparse
from pathlib import Path
import pandas as pd
import pyterrier as pt

INDEX_DIR = Path("data/pt_index_v2_month")
QUERIES_TXT = Path("data/queries.txt")

RUN_OUT = Path("runs/pt_bm25_test.jsonl")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    idx = pt.IndexRef.of(str(INDEX_DIR.resolve()))
    queries = load_queries_txt(QUERIES_TXT)

    retr = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=args.topk,
        metadata=["docno", "year", "month", "yyyymm", "doclen", "title_len", "abs_len"],
    )

    df = retr.transform(queries)
    df = df.sort_values(["qid", "rank"]).copy()
    df["rank"] = df.groupby("qid").cumcount()

    write_run_jsonl(df[["qid", "docno", "rank", "score"]], RUN_OUT)
    print(f"[BM25_V2] Run geschrieben: {RUN_OUT.resolve()} (TOPK={args.topk})")
    print(df.head())


if __name__ == "__main__":
    main()
