# TIC-TAC-TOE — Ausgabewert-Vergleich

Tic-Tac-Toe vs. MinMax. **9 Features** = Felder (0,0) … (2,2).

Charakteristik: **local_sverl** | Train: 100,000

**Erklärter Zustand (Brett):**

```
0 0 0
0 1 0
2 0 2
```
(0=leer, 1=Agent, 2=Gegner)

## Gesamtvergleich (pro Feld)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| (0,0) | 0.009 | 0.019 | 0.144 | 0.012 | 0.000 | 0.004 |
| (0,1) | 0.074 | 0.117 | 0.015 | 0.012 | 0.000 | 0.004 |
| (0,2) | 0.013 | 0.025 | 0.000 | 0.012 | 0.000 | 0.004 |
| (1,0) | 0.007 | 0.016 | 0.000 | 0.012 | 0.000 | 0.004 |
| (1,1) | 0.037 | 0.062 | 0.000 | 0.012 | 0.000 | 0.000 |
| (1,2) | 0.006 | 0.016 | 0.000 | 0.012 | 0.000 | 0.004 |
| (2,0) | 0.291 | 0.329 | 0.149 | 0.382 | 0.423 | 0.421 |
| (2,1) | 0.116 | 0.151 | 0.015 | 0.012 | 0.000 | -0.015 |
| (2,2) | 0.279 | 0.315 | 0.510 | 0.369 | 0.409 | 0.408 |

![ttt](vergleich_gesamt.png)

## Detailtabelle

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (0,0) | 0.009 | 0.019 | 0.144 | 0.012 | 0.000 | 0.004 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (0,1) | 0.074 | 0.117 | 0.015 | 0.012 | 0.000 | 0.004 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (0,2) | 0.013 | 0.025 | 0.000 | 0.012 | 0.000 | 0.004 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (1,0) | 0.007 | 0.016 | 0.000 | 0.012 | 0.000 | 0.004 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (1,1) | 0.037 | 0.062 | 0.000 | 0.012 | 0.000 | 0.000 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (1,2) | 0.006 | 0.016 | 0.000 | 0.012 | 0.000 | 0.004 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (2,0) | 0.291 | 0.329 | 0.149 | 0.382 | 0.423 | 0.421 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (2,1) | 0.116 | 0.151 | 0.015 | 0.012 | 0.000 | -0.015 |
| `(0, 0, 0, 0, 1, 0, 2, 0, 2)` | (2,2) | 0.279 | 0.315 | 0.510 | 0.369 | 0.409 | 0.408 |


Umgebung: Tic-Tac-Toe vs. MinMax. 9 Features = Brettfelder (0,0)…(2,2). Erklärter Zustand: Agent in der Mitte, Gegner in (2,0) und (2,2).

Befund: Nur die Felder (2,0) und (2,2) haben hohe Werte (~0,28–0,42) – die Positionen des Gegners. Shapley identifiziert beide klar (0,291 / 0,279). Banzhaf bestätigt mit etwas höheren Absolutwerten (0,329 / 0,315).

Methodenprobleme bei TTT:
•	Nucleolus: Feld (2,2) stark verzerrt auf 0,51 – überschätzt einen Gegner, ignoriert den anderen
•	Utopia: Setzt alle leeren Felder auf 0 – unterschlägt echte Beiträge (z. B. Feld (0,1) = 0,074 bei Shapley)
•	Gately: Ähnlich Utopia, überschätzt Schlüsselfelder, gibt Feld (2,1) sogar −0,015
•	Tau: Gleichmäßig 0,012 für viele Felder – kaum Differenzierung

✓  Empfehlung: Shapley. Banzhaf als Bestätigung. Nucleolus, Utopia, Gately und Tau bei 9-Feature-Spielen ungeeignet.
