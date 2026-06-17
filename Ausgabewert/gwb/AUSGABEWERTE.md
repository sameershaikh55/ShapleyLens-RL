# GWB — Ausgabewert-Vergleich


Hier haben die Features echten Einfluss; x ist wichtiger als y. Die meisten Methoden stimmen überein, Gately und Utopia weichen ab.

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|------
| x       | 1.147   | 1.147   | 1.147   | 1.107   | 1.571 | 4.069 |
| y       | 0.509   | 0.509   | 0.509   | 0.549   | 0.000 | 3.431 |

![gwb gesamt](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|------
| `(0, 0)` | x      | 4.088   | 4.088   | 4.088   | 4.088 | 6.114 | 4.031 |
| `(0, 0)` | y      | 2.026   | 2.026   | 2.026   | 2.026 | 0.000 | 1.969 |
| `(1, 0)` | x      | 0.329   | 0.329   | 0.329   | 0.169 | 0.169 | 3.744 |
| `(1, 0)` | y      | -0.160  | -0.160  | -0.160  | 0.000 | 0.000 | 3.256 |
| `(1, 1)` | x      | 0.090   | 0.090   | 0.090   | 0.090 | 0.000 | 4.000 |
| `(1, 1)` | y      | 0.090   | 0.090   | 0.090   | 0.090 | 0.000 | 4.000 |
| `(1, 2)` | x      | 0.082   | 0.082   | 0.082   | 0.082 | 0.000 | 4.500 |
| `(1, 2)` | y      | 0.082   | 0.082   | 0.082   | 0.082 | 0.000 | 4.500 |

Wichtigstes Feature (Mittel): x (Zeile) — 1.15 vs. 0.51 für y.

Methoden: Shapley ≈ Banzhaf ≈ Nucleolus am stabilsten. Tau ähnlich. Utopia weicht ab (y oft 0). Gately größere Zahlen, gleiche Tendenz (x > y).

Warum: Am Start (0,0) dominiert x (≈ 4.09) — die Zeile bestimmt oft den Weg zum Ziel; das Hindernis macht die Spalte weniger entscheidend.