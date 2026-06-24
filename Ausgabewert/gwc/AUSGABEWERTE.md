# GWC — Ausgabewert-Vergleich

2×4 Grid, zwei Hindernisse, 5 Zustände.

Charakteristik: **local_sverl** | Train: 300,000 | Rolls: 5,000

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| x | 1.143 | 1.143 | 1.143 | 1.156 | 1.152 | 1.143 |
| y | 1.553 | 1.553 | 1.553 | 1.541 | 1.545 | 1.553 |

![gwc gesamt](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | x | 4.000 | 4.000 | 4.000 | 4.000 | 3.978 | 4.000 |
| `(0, 0)` | y | 4.044 | 4.044 | 4.044 | 4.044 | 4.066 | 4.044 |
| `(1, 0)` | x | 0.134 | 0.134 | 0.134 | 0.193 | 0.258 | 0.134 |
| `(1, 0)` | y | 0.687 | 0.687 | 0.687 | 0.629 | 0.563 | 0.687 |
| `(1, 1)` | x | -0.002 | -0.002 | -0.002 | 0.000 | 0.000 | -0.002 |
| `(1, 1)` | y | 0.504 | 0.504 | 0.504 | 0.502 | 0.502 | 0.504 |
| `(1, 2)` | x | 1.255 | 1.255 | 1.255 | 1.255 | 1.191 | 1.255 |
| `(1, 2)` | y | 2.197 | 2.197 | 2.197 | 2.197 | 2.260 | 2.197 |
| `(0, 2)` | x | 0.328 | 0.328 | 0.328 | 0.330 | 0.330 | 0.328 |
| `(0, 2)` | y | 0.334 | 0.334 | 0.334 | 0.332 | 0.332 | 0.334 |


Umgebung: 2×4 Grid mit zwei Hindernissen, 5 Zustände. Features: x, y.

Befund: Umgekehrte Rangfolge zu GWB: y (~1,55) dominiert über x (~1,14). Zwei Hindernisse machen die vertikale Bewegung wichtiger. Alle Methoden (Shapley bis Gately) stimmen überein – Tau und Utopia weichen nur minimal ab (< 2 %).

✓  Empfehlung: Shapley (Banzhaf/Nucleolus/Gately identisch). Tau/Utopia akzeptabel, aber ohne Mehrwert.
