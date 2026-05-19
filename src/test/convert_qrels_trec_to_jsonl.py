import json
from pathlib import Path

QRELS_TREC = Path("data/qrels.txt")
QRELS_JSONL = Path("data/qrels.jsonl")

def main():
    n = 0
    with QRELS_TREC.open("r", encoding="utf-8") as fin, QRELS_JSONL.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Unerwartetes Format in Zeile {line_no}: {line}")

            qid = parts[0]
            doc_id = parts[2]
            rel = int(parts[3])

            fout.write(
                json.dumps({
                    "qid": str(qid),
                    "doc_id": str(doc_id),
                    "rel": rel
                }) + "\n"
            )
            n += 1

    print(f"✔ qrels.jsonl geschrieben: {QRELS_JSONL.resolve()} ({n} Einträge)")

if __name__ == "__main__":
    main()
