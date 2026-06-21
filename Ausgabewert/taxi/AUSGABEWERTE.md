# TAXI — Ausgabewert-Vergleich

Gymnasium **Taxi-v3** (faktorisierter Zustand). **4 Features**: taxi_row, taxi_col, passenger_loc, destination.

Value Iteration (γ=0.99) | Charakteristik: **local_sverl** | Rolls: 300

**Erklärte Zustände:**

- `(0, 0, 0, 1)` — Vor Abholung (an R)
- `(0, 0, 4, 1)` — Nach Abholung (an R, Passagier im Taxi)
- `(0, 4, 4, 1)` — Vor Absetzen (an G, Passagier im Taxi)
- `(2, 2, 2, 3)` — Generische Navigation (Mitte des Gitters)

## Gesamtvergleich (Mittel über alle vier Zustände)



![taxi](vergleich_gesamt.png)

| Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| taxi_row | 2.661 | 2.288 | 2.377 | 3.817 | 4.597 | 3.804 |
| taxi_col | 3.797 | 4.346 | 3.474 | 3.543 | 2.865 | 2.849 |
| passenger_loc | 8.131 | 8.546 | 6.696 | 5.003 | 5.743 | 8.388 |
| destination | 1.187 | 1.410 | 3.228 | 3.412 | 2.570 | 0.736 |

## Detailtabelle (pro Zustand)

| Zustand | Feature | Shapley | Banzhaf | Nucleolus | Tau | Utopia | Gately |
|:--------|:-------:|--------:|--------:|--------:|--------:|--------:|--------:|
| `(0, 0, 0, 1)` | taxi_row | 2.636 | 1.368 | 4.397 | 6.823 | 8.691 | 6.365 |
| `(0, 0, 0, 1)` | taxi_col | 7.512 | 9.716 | 10.073 | 6.067 | 3.619 | 3.891 |
| `(0, 0, 0, 1)` | passenger_loc | 15.686 | 17.649 | 11.760 | 7.567 | 13.673 | 14.747 |
| `(0, 0, 0, 1)` | destination | 0.149 | -0.151 | -0.247 | 5.527 | 0.000 | 0.980 |
| `(0, 0, 4, 1)` | taxi_row | -1.626 | -1.983 | -0.332 | 0.079 | 0.175 | -1.623 |
| `(0, 0, 4, 1)` | taxi_col | -0.612 | -1.174 | -0.462 | 0.212 | 0.472 | -0.198 |
| `(0, 0, 4, 1)` | passenger_loc | 4.702 | 4.143 | 2.823 | 2.342 | 2.110 | 4.083 |
| `(0, 0, 4, 1)` | destination | 0.763 | 0.477 | 1.198 | 0.594 | 0.469 | 0.965 |
| `(0, 4, 4, 1)` | taxi_row | 6.565 | 6.863 | 2.047 | 6.730 | 8.502 | 7.775 |
| `(0, 4, 4, 1)` | taxi_col | 5.296 | 5.960 | 0.803 | 6.170 | 6.217 | 5.118 |
| `(0, 4, 4, 1)` | passenger_loc | 8.314 | 8.810 | 10.088 | 5.117 | 0.370 | 10.378 |
| `(0, 4, 4, 1)` | destination | 4.725 | 6.226 | 11.962 | 6.883 | 9.812 | 1.629 |
| `(2, 2, 2, 3)` | taxi_row | 3.069 | 2.906 | 3.397 | 1.637 | 1.020 | 2.698 |
| `(2, 2, 2, 3)` | taxi_col | 2.993 | 2.882 | 3.483 | 1.724 | 1.152 | 2.584 |
| `(2, 2, 2, 3)` | passenger_loc | 3.822 | 3.581 | 2.113 | 4.988 | 6.821 | 4.343 |
| `(2, 2, 2, 3)` | destination | -0.891 | -0.911 | 0.000 | 0.644 | 0.000 | -0.633 |
