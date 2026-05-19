import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DOC_DIR = Path("data/documents")
CACHE_DB = Path("data/doc_text_cache.sqlite")


def _iter_jsonl_files(doc_dir: Path) -> Iterable[Path]:
    files = sorted(doc_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Keine .jsonl Dateien gefunden in: {doc_dir.resolve()}")
    return files


def _build_needed_docno_set(cands: pd.DataFrame) -> Set[str]:
    # expects a column 'docno' (PyTerrier output) or 'doc_id'
    if "docno" in cands.columns:
        return set(cands["docno"].astype(str).unique())
    if "doc_id" in cands.columns:
        return set(cands["doc_id"].astype(str).unique())
    raise KeyError("Candidates DF braucht Spalte 'docno' oder 'doc_id'.")


def _ensure_cache_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_text (
            docno TEXT PRIMARY KEY,
            text  TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_text_docno ON doc_text(docno)")
    conn.commit()


def _cache_has(conn: sqlite3.Connection, docno: str) -> bool:
    cur = conn.execute("SELECT 1 FROM doc_text WHERE docno=? LIMIT 1", (docno,))
    return cur.fetchone() is not None


def _cache_get_many(conn: sqlite3.Connection, docnos: List[str]) -> Dict[str, str]:
    # SQLite has a parameter limit, chunk it
    out: Dict[str, str] = {}
    chunk_size = 900  # safe under 999
    for i in range(0, len(docnos), chunk_size):
        chunk = docnos[i : i + chunk_size]
        qmarks = ",".join(["?"] * len(chunk))
        cur = conn.execute(f"SELECT docno, text FROM doc_text WHERE docno IN ({qmarks})", chunk)
        for docno, text in cur.fetchall():
            out[str(docno)] = text
    return out


def _cache_insert_many(conn: sqlite3.Connection, rows: List[Tuple[str, str]]) -> None:
    conn.executemany("INSERT OR IGNORE INTO doc_text(docno, text) VALUES(?, ?)", rows)
    conn.commit()


def build_doc_text_cache_for_candidates(
    candidates: pd.DataFrame,
    doc_dir: Path = DOC_DIR,
    cache_db: Path = CACHE_DB,
    verbose: bool = True,
) -> None:
    """
    Baut/füllt SQLite-Cache mit docno -> (title+abstract) Text,
    aber nur für docnos, die in 'candidates' vorkommen.
    Scannt die großen JSONLs 1x und speichert nur benötigte Docs.
    """
    needed = _build_needed_docno_set(candidates)
    if verbose:
        print(f"DOC_CACHE: benötigte docnos (unique): {len(needed)}")

    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_db))
    try:
        _ensure_cache_schema(conn)

        # Prüfe wie viele bereits drin sind (sample-based quick check)
        # (keine Vollzählung, das wäre teuer)
        already = 0
        sample = list(needed)[:2000]
        for d in sample:
            if _cache_has(conn, d):
                already += 1
        if verbose:
            print(f"DOC_CACHE: sample already in cache: {already}/{len(sample)}")

        # Wir scannen die JSONLs und sammeln Inserts in Batches
        pending: List[Tuple[str, str]] = []
        inserted = 0
        seen = 0

        for fp in _iter_jsonl_files(doc_dir):
            if verbose:
                print(f"DOC_CACHE: scan {fp.name} ...")
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    docno = str(obj.get("id", "") or "").strip()
                    if not docno or docno not in needed:
                        continue

                    title = str(obj.get("title", "") or "").strip()
                    abstract = str(obj.get("abstract", "") or "").strip()
                    text = (title + "\n\n" + abstract).strip()
                    if not text:
                        continue

                    pending.append((docno, text))
                    seen += 1

                    if len(pending) >= 2000:
                        _cache_insert_many(conn, pending)
                        inserted += len(pending)
                        pending = []
                        if verbose and inserted % 20000 == 0:
                            print(f"DOC_CACHE: inserted ~{inserted}")

        if pending:
            _cache_insert_many(conn, pending)
            inserted += len(pending)

        if verbose:
            print(f"DOC_CACHE: done. inserted rows (attempted): {inserted} | matched lines: {seen}")

    finally:
        conn.close()


def add_dense_feature(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_db: Path = CACHE_DB,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Erwartet:
      - candidates: DataFrame mit Spalten mindestens ['qid', 'docno', 'score', ...] (PyTerrier)
      - queries: DataFrame mit ['qid','query']

    Gibt candidates mit zusätzlicher Spalte 'f_dense' zurück.
    """
    if "qid" not in candidates.columns:
        raise KeyError("candidates braucht Spalte 'qid'")
    if "docno" not in candidates.columns and "doc_id" not in candidates.columns:
        raise KeyError("candidates braucht 'docno' (PyTerrier) oder 'doc_id'")
    if "qid" not in queries.columns or "query" not in queries.columns:
        raise KeyError("queries braucht Spalten ['qid','query']")

    # Normalisiere docno-Spalte
    cands = candidates.copy()
    if "docno" not in cands.columns:
        cands["docno"] = cands["doc_id"].astype(str)
    else:
        cands["docno"] = cands["docno"].astype(str)

    qmap = dict(zip(queries["qid"].astype(str), queries["query"].astype(str)))
    qids = cands["qid"].astype(str).unique().tolist()

    if verbose:
        print(f"DENSE: model={model_name}")
        print(f"DENSE: qids in candidates = {len(qids)} | rows={len(cands)}")

    # Load model
    model = SentenceTransformer(model_name)

    # Connect cache
    conn = sqlite3.connect(str(cache_db))
    try:
        _ensure_cache_schema(conn)

        # Compute dense per query (streaming)
        dense_scores = np.zeros(len(cands), dtype=np.float32)

        # group indices per query
        grouped = cands.groupby("qid", sort=False).indices  # qid -> row indices
        for qi, qid in enumerate(qids, start=1):
            idxs = grouped.get(qid, None)
            if idxs is None:
                continue

            query_text = qmap.get(qid, "")
            if not query_text:
                # unknown query => keep zeros
                continue

            docnos = cands.loc[idxs, "docno"].tolist()

            # fetch texts from cache
            text_map = _cache_get_many(conn, docnos)

            # build texts in same order; missing => empty
            texts = [text_map.get(d, "") for d in docnos]

            # if many missing: warn once in a while
            if verbose and qi % 50 == 0:
                missing = sum(1 for t in texts if not t)
                print(f"DENSE: {qi}/{len(qids)} qids done | missing texts this qid: {missing}/{len(texts)}")

            # encode (normalize embeddings => dot == cosine)
            q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
            d_emb = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # cosine similarity
            sims = (d_emb @ q_emb[0]).astype(np.float32)
            dense_scores[idxs] = sims

        cands["f_dense"] = dense_scores
        return cands

    finally:
        conn.close()
