# GWA — Ausgabewert-Vergleich

2×3 Grid, Ziel oben, 4 Zustände.

Charakteristik: **local_sverl** | Train: 300,000 | Rolls: 5,000

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

![gwa gesamt](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 0)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 1)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 1)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 0)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 0)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 1)` | x | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 1)` | y | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |



Umgebung: 2×3 Grid, Ziel oben, 4 Zustände. Features: x, y.

Befund: Alle Ausgabewerte für x und y sind exakt 0. Das Grid ist klein und symmetrisch; der optimale Pfad hängt nicht von der Trennung der Koordinaten ab. Die Ausgabewert-Analyse liefert hier keine Unterscheidungskraft – das ist kein Fehler, sondern korrekt.

✓  Empfehlung: Beliebige Methode – Ergebnis identisch (0). Kein interpretierbarer Feature-Beitrag vorhanden.
