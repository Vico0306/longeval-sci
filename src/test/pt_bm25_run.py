from pathlib import Path
import pyterrier as pt
import pandas as pd

index_path = Path("data/pt_index").resolve()
index_ref = pt.IndexRef.of(str(index_path))



TOPK = 1000  # Candidate Pool
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=TOPK)

def read_queries(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                qid, query = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    qid, query = parts[0], parts[1]
                else:
                    qid, query = str(i), line

            rows.append({"qid": str(qid).strip(), "query": str(query).strip()})

    if not rows:
        raise ValueError(f"queries.txt ist leer: {path.resolve()}")
    return pd.DataFrame(rows)


def main():
    # Java starten (ohne deprecated pt.init())
    if not pt.java.started():
        pt.java.init()

    index_path = INDEX_DIR.resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Index nicht gefunden: {index_path} (erst pt_index.py laufen lassen)")

    topics = read_queries(QUERIES_PATH.resolve())
    print(f"PT_BM25_RUN: Queries geladen: {len(topics)}")

    # Neuer API-Name statt BatchRetrieve:
    # pt.terrier.Retriever ist der Ersatz (deprecation fix)
    bm25 = pt.terrier.Retriever(str(index_path), wmodel="BM25", num_results=TOPK)

    print("PT_BM25_RUN: Starte BM25 Retrieval...")
    res = bm25.transform(topics)

    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RUN_PATH.open("w", encoding="utf-8") as f:
        for row in res.itertuples(index=False):
            f.write(
                f'{{"qid":"{row.qid}","doc_id":"{row.docno}","rank":{int(row.rank)},"score":{float(row.score)}}}\n'
            )

    print(f"PT_BM25_RUN: Run geschrieben nach: {RUN_PATH.resolve()} (TOPK={TOPK})")
    print(res.head(5))


if __name__ == "__main__":
    main()
