import json
import re
from pathlib import Path
import pyterrier as pt

DOC_DIR = Path("data/documents")
INDEX_DIR = Path("data/pt_index")


def token_count(text: str) -> int:
    # einfache robuste Tokenisierung
    return len(re.findall(r"\w+", text.lower()))


def iter_core_jsonl_docs(doc_dir: Path):
    """
    Erwartet .jsonl Dateien in data/documents/.
    Jede Zeile ist JSON und enthält:
      - id (string/int)
      - title (optional)
      - abstract (optional)
      - publishedDate (optional, ISO-like)
    Indexiert:
      - docno=id
      - text=title+abstract
      - meta: year, doclen, title_len, abs_len
    """
    jsonl_files = sorted(doc_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Keine .jsonl Dateien gefunden in: {doc_dir.resolve()}")

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON Fehler in {fp.name}:{line_no}: {e}")

                if "id" not in obj or obj["id"] is None:
                    raise KeyError(f"Fehlendes Feld 'id' in {fp.name}:{line_no}")

                docno = str(obj["id"])

                title = str(obj.get("title", "") or "").strip()
                abstract = str(obj.get("abstract", "") or "").strip()
                text = (title + "\n\n" + abstract).strip()

                # skip komplett leere Docs
                if not text:
                    continue

                published_date = str(obj.get("publishedDate", "") or "")
                year = ""
                if len(published_date) >= 4 and published_date[:4].isdigit():
                    year = published_date[:4]

                # Längenfeatures
                doclen = token_count(text)
                title_len = token_count(title) if title else 0
                abs_len = token_count(abstract) if abstract else 0

                yield {
                    "docno": docno,
                    "text": text,
                    "year": year,
                    "doclen": str(doclen),
                    "title_len": str(title_len),
                    "abs_len": str(abs_len),
                }


def main():
    if not pt.java.started():
        pt.java.init()

    index_path = INDEX_DIR.resolve()
    index_path.mkdir(parents=True, exist_ok=True)

    # Schreibtest
    testfile = index_path / "_write_test.tmp"
    testfile.write_text("ok", encoding="utf-8")
    testfile.unlink()

    print(f"Index baue ich in: {index_path}")

    # meta-Feldlängen: Strings, daher großzügig dimensionieren
    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={
            "docno": 40,
            "year": 4,
            "doclen": 10,
            "title_len": 10,
            "abs_len": 10,
        },
        text_attrs=["text"],
        overwrite=True
    )

    index_ref = indexer.index(iter_core_jsonl_docs(DOC_DIR))
    print("Fertig! IndexRef:", index_ref)


if __name__ == "__main__":
    main()
