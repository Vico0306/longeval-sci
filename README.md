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
