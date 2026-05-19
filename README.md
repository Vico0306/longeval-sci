# LongEval-Sci Retrieval Pipeline

Dieses Repository enthält unsere finale Retrieval-Pipeline für **LongEval-Sci / Task 1: Scientific Retrieval**.

Ziel ist es, wissenschaftliche Dokumente für Suchanfragen zu ranken und BM25 durch zusätzliche semantische, zeitliche und citation-basierte Features zu verbessern.

## Ansatz

Unser System nutzt eine hybride Pipeline:

1. **BM25** erzeugt Top-K Kandidaten  
2. Zusätzliche Features werden berechnet:
   - BM25 Score
   - Query-Länge
   - Dokumentlänge
   - Recency
   - Query-Dokument-Overlap
   - Title Match
   - Dense Similarity
   - Citation Count
3. Ein **LightGBM LambdaRank** Modell kombiniert diese Features
4. Das finale Ranking wird als Run-Datei ausgegeben

Die finale Score-Kombination basiert auf:
```text
final_score = BM25 + α · LTR
```

## Features

Unser Learning-to-Rank-Modell kombiniert verschiedene Feature-Gruppen, um die Relevanz wissenschaftlicher Dokumente besser einschätzen zu können.

| Feature | Gruppe | Beschreibung |
|---|---|---|
| `f_bm25` | Lexikalisch | BM25-Score des Dokuments für die jeweilige Query |
| `f_qlen` | Query | Länge der Query |
| `f_doclen` | Dokument | Länge des Dokuments |
| `f_recency` | Zeit | Alter des Dokuments in Monaten |
| `f_overlap` | Query-Dokument | Anteil der Query-Terme, die im Dokument vorkommen |
| `f_title_match` | Query-Dokument | Anteil der Query-Terme, die im Titel vorkommen |
| `f_dense` | Semantisch | Cosine Similarity zwischen Query- und Dokument-Embedding |
| `f_log_citation_count` | Citation | Logarithmierte Anzahl der Zitationen eines Dokuments |

### Feature-Gruppen

- **Lexikalische Features:** erfassen klassische Wortüberlappung über BM25.
- **Semantische Features:** erfassen inhaltliche Ähnlichkeit über Dense Embeddings.
- **Zeitliche Features:** berücksichtigen das Alter eines wissenschaftlichen Dokuments.
- **Citation Features:** dienen als Signal für wissenschaftlichen Einfluss.
- **Query-Dokument-Features:** messen direkte Überschneidungen zwischen Query, Titel und Dokumenttext.
# Evaluation

Die Evaluation erfolgt mit nDCG@10.

Für Snapshot 1 lagen Qrels vor. Dort erzielte das hybride System bessere Ergebnisse als BM25:

| System         |    nDCG@10 |
| -------------- | ---------: |
| BM25           |     0.3076 |
| Hybrid α = 0.1 |     0.3522 |
| Hybrid α = 0.2 |     0.3644 |
| Hybrid α = 0.3 |     0.3750 |
| Hybrid α = 0.4 | **0.3845** |
| Hybrid α = 0.5 |     0.3830 |

# Finale Umsetzung

Die finale Version befindet sich im Jupyter Notebook:
```text
final_ltr_bm25_hybrid.ipynb
```
Das Notebook enthält:

- Laden der LongEval-Daten
- Indexierung mit PyTerrier
- BM25 Retrieval
- Feature Engineering
- Dense Feature Berechnung
- LightGBM LambdaRank Training
- Run-Erstellung für die Snapshots

## Output

Die finalen Runs werden im TREC-Format erzeugt:

```text
qid Q0 docid rank score run_name
```

Beispiel:

```text
00dea7cb26b0df14733b1aa2e48d4189 Q0 11012449 1 48.36624608124125 team_vico_ltr
```

Für die Abgabe wird folgende Ordnerstruktur verwendet:

```text
longeval_submission/
├── snapshot-1/
│   └── run.txt.gz
├── snapshot-2/
│   └── run.txt.gz
├── snapshot-3/
│   └── run.txt.gz
└── ir-metadata.yml
```

# Technologien
- Python
- Jupyter Notebook
- PyTerrier
- LightGBM
- SentenceTransformers
- ir_datasets_longeval
- pandas
- numpy

# Fazit

Das Projekt zeigt, dass BM25 durch semantische, zeitliche und citation-basierte Features in Kombination mit Learning-to-Rank verbessert werden kann. Besonders effektiv war die hybride Kombination aus BM25 und LTR.
