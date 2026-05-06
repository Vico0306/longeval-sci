Setup:
Wir haben ein BM25-Baseline-System, ein Dense-Modell (all-MiniLM-L6-v2) und ein Hybrid-System (gewichtete Kombination aus BM25- und Dense-Scores mit α = 0.6) implementiert.

Evaluation:
Für ein kleines, manuell annotiertes Testset (3 Queries, 5 Dokumente, einfache Relevanzlabels) haben wir nDCG@3 berechnet.

Ergebnis (Toy-Setup):
BM25 und Hybrid erreichen beide einen durchschnittlichen nDCG@3 von ca. 0.82.
→ Im kleinen Setup ist das Hybrid-System noch nicht klar besser, aber die vollständige Pipeline für einen systematischen Vergleich ist implementiert (Run-File + nDCG-Eval).

In einer ersten Parameterstudie haben wir das Gewicht α zwischen BM25 und Dense-Scores variiert (0.3, 0.5, 0.7).
Auf unserem Toy-Testset zeigte sich α = 0.5 mit einem nDCG@3 von 0.8863 als beste Konfiguration, während BM25-only und Hybrid mit α ≥ 0.6 bei etwa 0.82 lagen.
| System    | α   | nDCG@3 |
| --------- | --- | ------ |
| BM25-only | –   | 0.8213 |
| Hybrid    | 0.3 | 0.8623 |
| Hybrid    | 0.5 | 0.8863 |
| Hybrid    | 0.6 | 0.8213 |
| Hybrid    | 0.7 | 0.8213 |

Neue Evaluation (7 Queries):
| System | α   | nDCG@3 |
| ------ | --- | ------ |
| BM25   | –   | 0.7091 |
| Hybrid | 0.5 | 0.7557 |

Mit einer erweiterten Menge von 7 Queries zeigt sich, dass das Hybrid-System mit α = 0.5 im Schnitt ein höheres nDCG@3 (0.756) erreicht als die BM25-Baseline (0.709).
Besonders bei semantisch komplexeren Anfragen (z. B. Q3, Q4) profitiert das Hybrid-System von der Kombination aus sparscher und dichter Repräsentation.

## Implemented Features Snapshot 2026 1 / 04/2026

# Implemented Features

| Feature | Variable | Description | Purpose |
|---|---|---|---|
| BM25 Score | `f_bm25` | Original BM25 relevance score | Baseline retrieval signal |
| Query Length | `f_qlen` | Number of terms in the query | Handles short vs. long queries |
| Document Length | `f_doclen` | Number of words in the document | Learns document length preferences |
| Publication Year | `f_year` | Publication year extracted from metadata | Captures temporal relevance |

---

# Core Ranking Features

```python
feature_cols = [
    "f_bm25",
    "f_qlen",
    "f_doclen",
    "f_year"
]
```

---

# Evaluation Results (NDCG@10)

| Model | Alpha | NDCG@10 |
|---|---|---|
| BM25 Baseline | - | `0.0635` |
| Hybrid BM25 + LTR | `0.05` | `0.0676` |
| Hybrid BM25 + LTR | `0.10` | `0.0723` |
| Hybrid BM25 + LTR | `0.20` | `0.0733` |
| Hybrid BM25 + LTR | `0.30` | `0.0693` |

---

# Best Configuration

```text
Hybrid BM25 + LightGBM LambdaRank
alpha = 0.20
NDCG@10 = 0.0733
```

## Implemented Features Snapshot 2026 1 / 05/2026

| Feature | Variable | Description | Purpose |
|---|---|---|---|
| BM25 Score | `f_bm25` | Original BM25 relevance score | Baseline retrieval signal |
| Query Length | `f_qlen` | Number of terms in the query | Handles short vs. long queries |
| Document Length | `f_doclen` | Number of words in the document | Learns document length preferences |
| Recency | `f_recency` | Document age in months | Promotes newer scientific documents |
| Query-Document Overlap | `f_overlap` | Token overlap between query and document text | Measures lexical similarity |
| Title Match | `f_title_match` | Token overlap between query and document title | Captures strong title relevance |

---

# Core Ranking Features

```python
feature_cols = [
    "f_bm25",
    "f_qlen",
    "f_doclen",
    "f_recency",
    "f_overlap",
    "f_title_match"
]
```

---

# Feature Importance

| Feature | Importance |
|---|---|
| `f_recency` | `5673.04` |
| `f_bm25` | `3086.18` |
| `f_doclen` | `2183.91` |
| `f_overlap` | `1059.30` |
| `f_title_match` | `824.56` |
| `f_qlen` | `750.11` |

The results indicate that **document recency** is the strongest ranking signal for the LongEval task.

---

# Evaluation Results (NDCG@10)

| Model | Alpha | NDCG@10 |
|---|---|---|
| BM25 Baseline | - | `0.0635` |
| Hybrid BM25 + LTR | `0.05` | `0.0676` |
| Hybrid BM25 + LTR | `0.10` | `0.0723` |
| Hybrid BM25 + LTR | `0.20` | `0.0733` |
| Hybrid BM25 + LTR | `0.30` | `0.0693` |

---

# Best Configuration

```text
Hybrid BM25 + LightGBM LambdaRank
alpha = 0.20
NDCG@10 = 0.0733
```
