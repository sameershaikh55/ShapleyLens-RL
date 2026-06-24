# FROZEN LAKE — Ausgabewert-Vergleich

Gym **FrozenLake-v1** (4×4, slippery). Features: **row**, **col**.

Value Iteration (γ=0.99) | Charakteristik: **local_sverl** | Rolls: 5,000

**Erklärte Zustände:** 11 (nicht-terminal)

Das Gitter ist symmetrisch → row ≈ col an den meisten Feldern. Fast alle Zustände haben Ausgabewert 0; nur nahe dem Ziel **(3,2)** weichen die Werte ab. Shapley, Banzhaf, Nucleolus und Gately stimmen überein; Tau und Utopia können leicht abweichen.

## Gesamtvergleich (Mittel über alle Zustände)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| row | -0.000 | -0.000 | -0.000 | 0.001 | 0.001 | -0.000 |
| col | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 |

## Visualisierung

![Übersicht](vergleich_gesamt.png)

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 0)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 1)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 1)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 2)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 2)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 3)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(0, 3)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 0)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 0)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 2)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(1, 2)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 0)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 0)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 1)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 1)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 2)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(2, 2)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(3, 1)` | row | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(3, 1)` | col | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `(3, 2)` | row | -0.001 | -0.001 | -0.001 | 0.008 | 0.008 | -0.001 |
| `(3, 2)` | col | -0.006 | -0.006 | -0.006 | -0.014 | -0.014 | -0.006 |



Umgebung: Gym FrozenLake-v1 (4×4, slippery). Features: row, col. Value Iteration (γ=0.99), 11 nicht-terminale Zustände, 5.000 Monte-Carlo-Rolls.

Befund: Alle Ausgabewerte liegen nahe 0. Das Gitter ist symmetrisch – kein Feature trägt messbar zum Return bei. Shapley, Banzhaf, Nucleolus und Gately sind identisch; Tau und Utopia weichen am Zielzustand minimal ab (row ≈ +0,001).

✓  Empfehlung: Shapley (oder Banzhaf/Nucleolus/Gately – alle identisch). Tau/Utopia bringen keinen Mehrwert bei symmetrischen Spielen.
