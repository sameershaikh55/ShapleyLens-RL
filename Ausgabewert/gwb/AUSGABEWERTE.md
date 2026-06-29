# GWB — Ausgabewert-Vergleich

2×4 Grid, Hindernis bei (0,1), 4 Zustände.

Charakteristik: **local_sverl** | Train: 300,000 | Rolls: 5,000

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| x | 1.146 | 1.146 | 1.146 | 1.104 | 1.575 | 1.146 |
| y | 0.512 | 0.512 | 0.512 | 0.553 | 0.000 | 0.512 |

![gwb gesamt](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | x | 4.080 | 4.080 | 4.081 | 4.081 | 6.129 | 4.080 |
| `(0, 0)` | y | 2.048 | 2.048 | 2.048 | 2.048 | 0.000 | 2.048 |
| `(1, 0)` | x | 0.339 | 0.339 | 0.339 | 0.173 | 0.173 | 0.339 |
| `(1, 0)` | y | -0.166 | -0.166 | -0.166 | 0.000 | 0.000 | -0.166 |
| `(1, 1)` | x | 0.078 | 0.078 | 0.078 | 0.078 | 0.000 | 0.078 |
| `(1, 1)` | y | 0.078 | 0.078 | 0.078 | 0.078 | 0.000 | 0.078 |
| `(1, 2)` | x | 0.087 | 0.087 | 0.087 | 0.087 | 0.000 | 0.087 |
| `(1, 2)` | y | 0.087 | 0.087 | 0.087 | 0.087 | 0.000 | 0.087 |

Umgebung: 2×4 Grid mit Hindernis bei (0,1), 4 Zustände. Features: x, y.

Befund: Erstmals klare Unterscheidung: x (horizontale Koordinate) ≈ 1,15 ist wichtiger als y ≈ 0,51. Das Hindernis blockiert die y-Richtung stärker. Shapley, Banzhaf, Nucleolus und Gately überlappen exakt. Utopia verzerrt: x auf 1,58 überschätzt, y auf 0 gesetzt. Tau leicht abweichend (~1,10 / ~0,55).

Kernbefund: x › y. Utopia ungeeignet (setzt y = 0). Tau brauchbar, aber leichte Abweichung.

✓  Empfehlung: Shapley. Banzhaf/Nucleolus/Gately gleichwertig. Utopia und Tau vermeiden.
