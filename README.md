# CGT_REINFORCEMENT

Repository für das Projekt **„Cooperative game solutions for reinforcement learning and other machine learning applications“**.

Das Projekt erklärt die Entscheidungen trainierter Reinforcement-Learning-Agenten (Q-Learning bzw. Value Iteration), indem der Beitrag jedes einzelnen Zustands-Features mit Methoden aus der kooperativen Spieltheorie berechnet wird: **Shapley-Wert, Banzhaf-Wert, Nucleolus, Tau-Value, Utopia-Payoff und Gately-Value**. Die Basis-Methodik (SVERL – *Shapley Values for Explaining Reinforcement Learning*) stammt aus Beechey, Smith & Şimşek, ICML 2023 ([Paper](https://arxiv.org/abs/2306.05810)); dieses Repository portiert und erweitert den Code (Umstieg von `gym` auf `gymnasium`, NumPy 2, neue Umgebungen, zusätzliche Auswertungsmethoden).

---

## Inhaltsverzeichnis

1. [Projektstruktur](#projektstruktur)
2. [Installation](#installation)
3. [Python-Version](#python-version)
4. [Benötigte Pakete](#benötigte-pakete)
5. [Experimente ausführen](#experimente-ausführen)
6. [Ausgabe / Ergebnisse](#ausgabe--ergebnisse)
7. [Troubleshooting](#troubleshooting)
8. [Bekannte Einschränkungen](#bekannte-einschränkungen)
9. [Zitation](#zitation)

---

## Projektstruktur

```
CGT_REINFORCEMENT/
│
├── README.md                        # Diese Datei
├── requirements.txt                 # Python-Abhängigkeiten
│
├── characteristics.py               # Berechnung der Charakteristikfunktion v(C)
├── game_computation.py              # Zentrale Spielberechnung (π_C, v_C, ...)
├── explainer.py                     # Alternative/ältere Explainer-Pipeline (main.py)
├── shapley.py / banzhaf.py /
│   nucleolus.py / tau.py /
│   utopia_payoff.py / gately.py     # Die sechs Lösungskonzepte der kooperativen Spieltheorie
├── q_agent_1.py / q_agent_2.py      # Q-Learning-Agenten (q_agent_2: mit gültigen Aktionsmasken)
├── utils.py                         # Training, Value Iteration, State-Sampling, Hilfsfunktionen
├── result_analyzer.py /
│   result_calculator.py /
│   result_visualizer.py             # Nachgelagerte Auswertung & Plot-Erzeugung
├── printer.py / pkl.py              # Kleine Hilfsmodule (Ausgabe, Pickle-I/O)
│
├── gwa/ gwb/ gwc/ gwd/              # Grid-World-Umgebungen (jeweils *.py + run.py)
├── taxi/                            # Gymnasium Taxi-v3 (Wrapper: taxi_wrap.py)
├── tic_tac_toe/                     # Tic-Tac-Toe gegen MinMax-Gegner
├── minesweeper/                     # 4×4-Minesweeper mit 2 Minen
├── frozen_lake/                     # Gymnasium FrozenLake-v1
├── battleship/                      # Nur Ergebnis-Dateien (.pkl) enthalten
│
├── cache/                           # Zwischengespeicherte Berechnungen (*.cache, git-ignoriert)
├── outputs/                         # Trainings-/Rohausgaben einzelner Läufe (git-ignoriert)
│
├── PORTING_NOTES.md                 # Detaillierte Migrationshinweise (gym→gymnasium, NumPy 2, Multiprocessing)
└── README_copy.md                   # Original-README des Upstream-Projekts SVERL (ICML 2023)
```

---

## Installation

### Voraussetzungen

- **Git** (zum Klonen des Repositories)
- **Python 3.11 – 3.14** (siehe [Python-Version](#python-version))
- **pip** (wird mit Python mitinstalliert)

### 1. Repository klonen

```bash
git clone https://github.com/jhstaudacher/CGT_REINFORCEMENT.git
cd CGT_REINFORCEMENT
```

### 2. Virtuelle Umgebung erstellen

Eine virtuelle Umgebung verhindert Konflikte mit anderen Python-Projekten auf dem System.

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

**Linux / macOS:**
```bash
python3 -m venv .venv
```

### 3. Virtuelle Umgebung aktivieren

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

> Nach erfolgreicher Aktivierung erscheint `(.venv)` vor dem Terminal-Prompt.
> Falls unter Windows die PowerShell-Ausführung von Skripten blockiert wird (`... cannot be loaded because running scripts is disabled ...`), einmalig als Administrator ausführen:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 5. Installation überprüfen

```bash
python --version
pip list
```

Erwartete Kernpakete in der Ausgabe von `pip list`: `numpy`, `scipy`, `gymnasium`, `matplotlib`, `tqdm`, `cloudpickle`, `colorama` (siehe [Benötigte Pakete](#benötigte-pakete)).

---

## Python-Version

**Getestet mit Python 3.12** (im Repository enthaltene `__pycache__`-Dateien wurden mit CPython 3.12 kompiliert). Die Migrationsdokumentation (`PORTING_NOTES.md`) beschreibt zusätzlich die Anpassungen für **Python 3.14**.

Minimal erforderlich ist **Python 3.11**, da dies die niedrigste Version ist, die von den gepinnten Abhängigkeiten unterstützt wird:

| Paket | Minimal unterstützte Python-Version |
|---|---|
| `numpy==2.4.4` | ≥ 3.11 |
| `gymnasium==1.2.3` | ≥ 3.10 |
| `matplotlib>=3.8.0` | ≥ 3.9 (aktuellste Version ≥ 3.11) |

**Empfehlung:** Python **3.11, 3.12, 3.13 oder 3.14**. Python 3.10 oder älter wird **nicht unterstützt** (NumPy 2.4.4 schlägt fehl).

Aktuelle Python-Version prüfen:

```bash
python --version
```

---

## Benötigte Pakete

Alle Abhängigkeiten sind in [`requirements.txt`](requirements.txt) gepinnt und werden mit `pip install -r requirements.txt` installiert.

| Paket | Version | Zweck |
|---|---|---|
| `numpy` | `==2.4.4` | Numerische Arrays, Q-Tables, Zustandsvektoren |
| `scipy` | `>=1.10.0` | `scipy.optimize.linprog` – wird für den Nucleolus (`nucleolus.py`) benötigt |
| `gymnasium` | `==1.2.3` | RL-Umgebungen `Taxi-v3` und `FrozenLake-v1` |
| `matplotlib` | `>=3.8.0` | Erzeugung der Vergleichsplots in `result_visualizer.py` |
| `tqdm` | `==4.67.3` | Fortschrittsbalken beim Training und bei der Koalitionsberechnung |
| `cloudpickle` | `==3.1.2` | Erweiterte Pickle-Unterstützung (Multiprocessing-Kompatibilität) |
| `colorama` | `==0.4.6` | Farbige Terminalausgabe unter Windows (u. a. für `tqdm`) |

Nur **Standardbibliothek** wird darüber hinaus verwendet (`argparse`, `pickle`, `pathlib`, `multiprocessing`, `itertools`, `math`, `copy`, `json`, `csv`, `importlib`, `inspect`, `typing`, u. a.) – hierfür ist keine zusätzliche Installation nötig.

> **Wichtiger Fund bei der Analyse:** `matplotlib` wird in `result_visualizer.py` importiert, war jedoch **nicht** in der ursprünglichen `requirements.txt` enthalten (diese war zudem versehentlich UTF-16-kodiert statt UTF-8, siehe [Troubleshooting](#troubleshooting)). Beides wurde in diesem Repository-Stand korrigiert: `matplotlib>=3.8.0` wurde ergänzt und die Datei als UTF-8 gespeichert.

---

## Experimente ausführen

Der empfohlene und zentrale Weg, um Experimente in diesem Projekt durchzuführen, ist die Nutzung der **`main.py`** (die auf der `Explainer`-Pipeline basiert). Hier können Sie alle gewünschten Spiele und Auswertungsmethoden an einem Ort konfigurieren und gesammelt ausführen.

### Zentrale Steuerung über `main.py`

Öffnen Sie die Datei `main.py` in einem Texteditor. Dort finden Sie die Initialisierung der `Explainer`-Klasse, die Sie nach Ihren Bedürfnissen anpassen können:

*   **`games`**: Eine Liste der Umgebungen, die ausgeführt werden sollen. Gültige Werte sind z. B. `"gwa"`, `"gwb"`, `"gwc"`, `"gwd"`, `"minesweeper"`, `"taxi"`, `"tic_tac_toe"`, `"frozen_lake"`, `"battleship"`.
*   **`explainers`**: Eine Liste der spieltheoretischen Methoden. Gültige Werte sind: `"shapley"`, `"banzhaf"`, `"nucleolus"`, `"tau"`, `"utopia-payoff"`, `"gately"`.
*   **`configs`**: Hier können Sie Tupel für direkte Vergleiche zwischen verschiedenen Explainern auf bestimmten Spielen definieren.
*   **`use_cache`** (`True`/`False`): Legt fest, ob bereits berechnete Charakteristikfunktionen von der Festplatte geladen werden sollen (setzt einen vorherigen Durchlauf voraus) oder ob alles neu berechnet wird.
*   **`normalize`** (`True`/`False`): Aktiviert die Normalisierung (wirkt sich nur auf `utopia-payoff`, `gately` und `banzhaf` aus).

Sobald Sie Ihre Konfiguration vorgenommen haben, führen Sie das Skript im Projekt-Root aus:

```bash
python main.py
```

Die Ergebnisse dieses zentralen Durchlaufs (inklusive JSON, CSV, Pickle-Dateien sowie generierter Heatmap-Plots und Vergleiche) werden automatisch im Ordner **`outputs/`** gespeichert.

---

### Alternative: Einzelne Umgebungen separat ausführen

Falls Sie ein spezifisches Experiment isoliert ausführen möchten, hat jede Umgebung weiterhin einen eigenen Ordner mit einer separaten `run.py`. Diese führt das Training (Q-Learning bzw. Value Iteration) **und** die Berechnung der Erklärwerte durch.

Die Ausführung erfolgt immer **aus dem jeweiligen Unterordner heraus**:

```bash
cd <ordnername>
python run.py
```

| Umgebung | Befehl | Ungefähre Laufzeit* | Bemerkung |
|---|---|---|---|
| Grid World A | `cd gwa && python run.py` | ~30 s | 2×3-Grid, 4 Zustände |
| Grid World B | `cd gwb && python run.py` | ~30 s | 2×4-Grid, 1 Hindernis |
| Grid World C | `cd gwc && python run.py` | ~30 s | 2×4-Grid, 2 Hindernisse |
| Grid World D | `cd gwd && python run.py` | ~1–2 min | 10×10-Grid, 20 Blöcke, **zufällig generiert** → Ergebnisse variieren pro Lauf |
| Taxi | `cd taxi && python run.py` | ~1–2 min | Gymnasium `Taxi-v3`, exakte Lösung per Value Iteration |
| FrozenLake | `cd frozen_lake && python run.py` | ~1–2 min | Gymnasium `FrozenLake-v1`, Value Iteration |
| Tic-Tac-Toe | `cd tic_tac_toe && python run.py` | ~5–15 min | 512 Koalitionen (9 Features) |
| Minesweeper | `cd minesweeper && python run.py` | ⚠️ sehr lang (ggf. Stunden) | 4×4-Feld, 16 Features → bis zu 32.768–65.536 Koalitionen (Nutzt `fast_local_sverl`) |

\* *Richtwerte auf einem üblichen Laptop; abhängig von CPU-Kernen (`multi_process`/`num_p`-Einstellungen in den jeweiligen `run.py`).*

Die Ergebnisse dieser isolierten Läufe werden als `.pkl`-Dateien direkt im jeweiligen Umgebungsordner gespeichert.

---

## Ausgabe / Ergebnisse

| Ordner | Inhalt |
|---|---|
| `<umgebung>/results.pkl` | Rohdaten (Shapley-Werte etc.) |
| `<umgebung>/*.pkl` (z. B. `shapley_shapley_on_value.pkl`) | Einzelergebnisse aus `run.py` |
| `cache/<umgebung>/*.cache` | Zwischengespeicherte Charakteristikfunktionswerte (beschleunigt erneute Läufe; git-ignoriert) |
| `outputs/` | Zusätzliche Trainings-Rohausgaben einzelner Experimente (git-ignoriert) |

---

## Troubleshooting

| Problem | Ursache | Lösung |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (o. ä.) | Virtuelle Umgebung nicht aktiviert | Schritt 3 der Installation wiederholen (`.venv` aktivieren) |
| `ModuleNotFoundError: No module named 'matplotlib'` | Alte/unvollständige `requirements.txt` | Aktuelle `requirements.txt` verwenden (`matplotlib` ist jetzt enthalten) und `pip install -r requirements.txt` erneut ausführen |
| `requirements.txt` lässt sich nicht öffnen / Zeichensalat | Datei war ursprünglich UTF-16-kodiert | Behoben – Datei ist jetzt UTF-8-kodiert |
| `RuntimeError` bezüglich Multiprocessing-„Bootstrapping“ beim Start von`run.py` | Fehlender `if __name__ == '__main__':`-Schutz (unter `spawn`/`forkserver`) | Alle `run.py`-Dateien in diesem Repository enthalten diesen Schutz bereits (siehe `PORTING_NOTES.md`, Abschnitt 5) |
| `_pickle.PicklingError` bei `multi_process=True` | Nicht picklebare `lambda`-Funktionen/`defaultdict` in Objektzustand | Bereits behoben (`characteristics.py`, `gwd/gwd.py`, `dict()`-Konvertierung in den `run.py`-Dateien) – siehe `PORTING_NOTES.md`, Abschnitt 6 |
| Minesweeper-Lauf dauert extrem lange / `MemoryError` | 2¹⁶ = 65.536 Koalitionen bei 16 Features | Mit `Ctrl+C` abbrechen |
| PowerShell: „running scripts is disabled“ beim Aktivieren der venv | Windows-Skriptausführung standardmäßig eingeschränkt | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` einmalig als Administrator ausführen |
| Ergebnisse bei Grid World D unterscheiden sich zwischen Läufen | `gwd` generiert ein zufälliges 10×10-Grid | Erwartetes Verhalten, kein Fehler |

---

## Bekannte Einschränkungen

- **Minesweeper** mit der Standard-`local_sverl`-Charakteristik (`minesweeper/run.py`) ist wegen 2¹⁶ Koalitionen praktisch nicht in vertretbarer Zeit berechenbar.
- Multiprocessing-Parameter (`multi_process`, `num_p`) sind in den `run.py`-Dateien teils hart auf hohe Prozesszahlen (z. B. `num_p=50` für Minesweeper) eingestellt und sollten je nach verfügbaren CPU-Kernen angepasst werden.

Weitere technische Details zur Portierung (Python 3.14, Gymnasium, NumPy 2, Multiprocessing/Pickling) sind in [`PORTING_NOTES.md`](PORTING_NOTES.md) dokumentiert.

---

## Zitation

Dieses Repository baut auf der SVERL-Methodik auf. Bei Verwendung bitte die Originalarbeit zitieren:

```bibtex
@inproceedings{beechey2023explaining,
  title={Explaining reinforcement learning with shapley values},
  author={Beechey, Daniel and Smith, Thomas MS and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r},
  booktitle={International Conference on Machine Learning},
  pages={2003--2014},
  year={2023},
  organization={PMLR}
}
```

Original-Repository und -README des Upstream-Projekts: siehe [`README_copy.md`](README_copy.md).
