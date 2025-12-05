Setup:
Wir haben ein BM25-Baseline-System, ein Dense-Modell (all-MiniLM-L6-v2) und ein Hybrid-System (gewichtete Kombination aus BM25- und Dense-Scores mit α = 0.6) implementiert.

Evaluation:
Für ein kleines, manuell annotiertes Testset (3 Queries, 5 Dokumente, einfache Relevanzlabels) haben wir nDCG@3 berechnet.

Ergebnis (Toy-Setup):
BM25 und Hybrid erreichen beide einen durchschnittlichen nDCG@3 von ca. 0.82.
→ Im kleinen Setup ist das Hybrid-System noch nicht klar besser, aber die vollständige Pipeline für einen systematischen Vergleich ist implementiert (Run-File + nDCG-Eval).
