# GWC — Ausgabewert-Vergleich

Hier haben beide Features Einfluss, wobei y im Durchschnitt wichtiger als x ist. Die Methoden Shapley, Banzhaf, Nucleolus und Tau stimmen weitgehend überein und zeigen ein stabiles Ranking, während Utopia leicht abweicht und Gately deutlich größere Werte vergibt, da es den Gesamtwert verteilt.

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| x | 1.113 | 1.113 | 1.113 | 1.128 | 1.126 | 3.297 |
| y | 1.542 | 1.542 | 1.542 | 1.527 | 1.529 | 3.703 |

![gwc](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | x | 3.927 | 3.927 | 3.927 | 3.927 | 3.920 | 2.518 |
| `(0, 0)` | y | 3.943 | 3.943 | 3.943 | 3.943 | 3.951 | 2.482 |
| `(1, 0)` | x | 0.163 | 0.163 | 0.163 | 0.218 | 0.279 | 2.742 |
| `(1, 0)` | y | 0.665 | 0.665 | 0.665 | 0.610 | 0.549 | 3.258 |
| `(1, 1)` | x | -0.005 | -0.005 | -0.005 | 0.000 | 0.000 | 3.255 |
| `(1, 1)` | y | 0.506 | 0.506 | 0.506 | 0.501 | 0.501 | 3.745 |
| `(1, 2)` | x | 1.171 | 1.171 | 1.171 | 1.171 | 1.107 | 3.457 |
| `(1, 2)` | y | 2.242 | 2.242 | 2.242 | 2.242 | 2.305 | 4.543 |
| `(0, 2)` | x | 0.311 | 0.311 | 0.311 | 0.326 | 0.326 | 4.514 |
| `(0, 2)` | y | 0.354 | 0.354 | 0.354 | 0.340 | 0.340 | 4.486 |

Wichtigstes Feature (Mittel): y (Spalte) — 1.54 vs. 1.11 für x.

Methoden: Shapley ≈ Banzhaf ≈ Nucleolus ≈ Tau — stabiles Ranking. Utopia leicht anders. Gately größere Zahlen, gleiche Tendenz (y > x).

Warum: Die Spalte entscheidet oft, wie man um die Hindernisse navigiert — besonders deutlich bei (1,2) (y ≈ 2.24, x ≈ 1.17).
