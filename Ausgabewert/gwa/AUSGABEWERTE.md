# GWA — Ausgabewert-Vergleich

Gately ist größer, weil es den Gesamtwert aufteilt und nicht den tatsächlichen Einfluss misst.

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.250 |
| y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.250 |

![gwa](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.000 |
| `(0, 0)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.000 |
| `(0, 1)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.500 |
| `(0, 1)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.500 |
| `(1, 0)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.000 |
| `(1, 0)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.000 |
| `(1, 1)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.500 |
| `(1, 1)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.500 |

Wichtigstes Feature: Keines — x ≈ y, kein Unterschied.

Methoden: Shapley, Banzhaf, Nucleolus, Tau, Utopia alle ≈ 0. Nur Gately zeigt Werte (~4.25) — andere Aufteilungslogik, kein echtes Feature-Ranking.

Warum: Das Gitter ist symmetrisch; row und col sind austauschbar → klassische Methoden sehen keinen marginalen Unterschied. Randfall — wie Frozen Lake, aber mit sichtbaren Gately-Werten.
