# Portierung: Python 3.14, Gymnasium und Multiprocessing

Diese Datei fasst die **Probleme** und **Änderungen** zusammen, die beim Umstieg auf eine aktuelle Python-Umgebung (insbesondere **Python 3.14**) und beim Betrieb mit **`multi_process=True`** relevant waren.

---

## 1. Ziel und Kontext

- **Python:** Zielversion z. B. **3.14.x** (neuere Interpreter setzen u. a. andere **Standard-Startmethoden** für `multiprocessing` voraus, siehe Abschnitt 5).
- **Abhängigkeiten:** Siehe [`requirements.txt`](requirements.txt) (u. a. **Gymnasium**, **NumPy 2.x**, **tqdm**, **colorama**).
- **Alt:** `gym` wurde durch **Gymnasium** ersetzt; **NumPy 1.x** + `np.math` sind mit **NumPy 2** nicht mehr kompatibel.

---

## 2. Abhängigkeiten (`requirements.txt`)

| Thema | Änderung |
|--------|----------|
| **Gym** | Entfernt; stattdessen **`gymnasium`** (API weitgehend kompatibel mit Gym 0.26). |
| **NumPy** | Version angehoben (z. B. **2.4.x**), passend zur Python-Version. |
| **cloudpickle** | Optional / oft unnötig, wenn im Code nicht importiert. |

---

## 3. NumPy 2 und Shapley (`shapley.py`)

- **`np.math.factorial`** existiert in **NumPy 2** nicht mehr.
- **Änderung:** `import math` und **`math.factorial(...)`** für die Shapley-Formel (siehe aktuelle [`shapley.py`](shapley.py)).

---

## 4. Taxi-Umgebung: Gymnasium und Beobachtung

- **`taxi/taxi_wrap.py` / `taxi/run.py`:** `import gymnasium as gym`, `gym.make('Taxi-v3')`, `ObservationWrapper` / Spaces von Gymnasium.
- **`utils.py`** nutzt bereits das **fünfteilige** `step`-Ergebnis (`terminated`, `truncated`) – kompatibel mit Gymnasium.

### `value_iteration` und Attribut `P`

- [`utils.value_iteration`](utils.py) erwartet **`env.P`** (Übergangsmatrix des **unverpackten** Taxi-Env).
- Ein **`ObservationWrapper`** (z. B. `FactoredState`) leitet **`P`** nicht automatisch weiter → **`AttributeError: ... has no attribute 'P'`**, falls nicht ergänzt.
- **Lösung:** Im Wrapper **`P`** vom inneren Env setzen, z. B. `self.P = self.env.unwrapped.P` (oder gleichwertig), oder `value_iteration` nur mit dem **Roh-Env** aufrufen und die faktorisierte Beobachtung nur dort nutzen, wo nötig.

---

## 5. Multiprocessing: `if __name__ == '__main__'`

- Unter **forkserver** / **spawn** (häufiger Standard unter neueren Python-Versionen auf Linux) wird das Startskript in Workerprozessen **erneut importiert**.
- **Ohne** Schutz führt **gesamter Top-Level-Code** (Training, `Manager()`, …) zu erneuten Starts → **`RuntimeError`** (Bootstrapping).
- **Änderung:** Ausführbarer Code in allen **`run.py`**-Skripten unter **`if __name__ == '__main__':`** (ggf. `main()` + Aufruf).

---

## 6. Pickling und `multi_process=True` (neu sichtbar geworden)

### 6.1 Warum erst „jetzt“?

- Früher war unter Linux oft **`fork`** die Voreinstellung; mit **`forkserver`** müssen u. a. **`Process`-Argumente** und der Zustand, der in Worker transportiert wird, **per Pickle** serialisierbar sein.
- **Lokale Lambdas**, **`defaultdict(lambda: …)`** und ähnliche Objekte sind oft **nicht** oder **schlecht** pickelbar → **`_pickle.PicklingError`**.

Die Logik war schon vorher riskant; sie fällt mit **forkserver** und **`multi_process=True`** **zuverlässig** auf.

### 6.2 `q_agent_1` und `pi_Cs` (`defaultdict` + `lambda`)

- `Agent.get_pi_C` baut **`defaultdict(lambda: np.full(...))`** – die **`lambda`** ist **nicht** pickelbar.
- **Pragmatische Lösung in den `run.py`:** Nach dem Erzeugen von `pi_Cs` innere Maps in normale **`dict`**-Objekte überführen, z. B.:

  ```python
  pi_Cs = {tuple(C): dict(agent.get_pi_C(...)) for C in ...}
  ```

  oder zweizeilig mit `{k: dict(v) for k, v in pi_Cs.items()}`.

- **Hinweis:** [`q_agent_2`](q_agent_2.py) nutzt **`PolicyDict(dict)`** mit **`__missing__`** – dort tritt dieses Problem typischerweise **nicht** in gleicher Form auf.

### 6.3 Global SVERL: `get_policy` (`characteristics.py`)

- **`global_sverl_C_values`** setzte früher **`self.get_policy = lambda state, C: ...`** – **Lambdas** im Objektzustand sind beim Pickeln von **`Characteristics`** problematisch.
- **Änderung:** Echte Methodenfunktion **`get_policy_global(self, state, C)`** und Zuweisung **`self.get_policy = self.get_policy_global`** (siehe aktuelle [`characteristics.py`](characteristics.py)).

### 6.4 `gwd.Grid`: `trans` (`defaultdict` + `lambda`)

- In [`gwd/gwd.py`](gwd/gwd.py) wurde **`self.trans`** mit **`defaultdict(lambda: ...)`** aufgebaut.
- Beim Multiprocessing wird **`env`** (also **`Grid`**) mit **`Characteristics`** serialisiert → derselbe Pickle-Fehler.
- **Änderung:** Nach dem Füllen aller Transitionen z. B. **`self.trans = dict(self.trans)`**, sodass kein unpickelbares **`default_factory`** mehr nötig ist.

---

## 7. Fortschrittsbalken: `16/16` vs. `2/2`

- **`multi_process=False`:** `tqdm` zählt **eine Iteration pro Koalition** → z. B. **16** Schritte bei 4 Merkmalen (alle Teilmengen).
- **`multi_process=True`:** Der äußere Balken zählt **`ceil(Anzahl_Koalitionen / num_p)`** Batches → z. B. **2** bei 16 Koalitionen und **`num_p=8`**.
- Die **Anzahl der berechneten Koalitionen** ist gleich; nur die **Anzeige** der Schritte ändert sich.

---

## 8. Betroffene Dateien (Überblick)

| Bereich | Dateien |
|---------|---------|
| Abhängigkeiten | [`requirements.txt`](requirements.txt) |
| Shapley / NumPy 2 | [`shapley.py`](shapley.py) |
| Taxi / Gymnasium | [`taxi/taxi_wrap.py`](taxi/taxi_wrap.py), [`taxi/run.py`](taxi/run.py) |
| Global SVERL / Pickle | [`characteristics.py`](characteristics.py) |
| `pi_Cs` als `dict` | u. a. [`gwa/run.py`](gwa/run.py), [`gwc/run.py`](gwc/run.py), [`taxi/run.py`](taxi/run.py) – alle `run.py`, die **`q_agent_1`** + **`multi_process=True`** nutzen |
| `gwd.trans` | [`gwd/gwd.py`](gwd/gwd.py) |
| Einstiegsskripte | alle **`*/run.py`** mit **`if __name__ == '__main__':`** |

---

## 9. Optionale weitere Aufräumarbeiten

- **`q_agent_1.py`:** `defaultdict(lambda: …)` in **`get_pi_C` / `get_policy`** dauerhaft durch **pickelbare** Fabriken (z. B. `functools.partial`) oder reine **`dict`** ersetzen – dann entfällt die **`dict(...)`**-Konvertierung in jedem `run.py`.
- **`FactoredState`:** Falls Gymnasium **Warnings** zu **`_observation_space`** meldet, auf die öffentliche **`observation_space`**-Zuweisung prüfen.
- **Minesweeper / `gwd`:** Koordinaten aus linearen Indizes bei **nicht quadratischen** Gittern gegen die **`reshape`**-Reihenfolge validieren (Logik, nicht nur Portierung).

---

*Stand: Dokumentation der im Projekt besprochenen und umgesetzten Anpassungen für Python 3.14, Gymnasium, NumPy 2 und Multiprocessing.*
