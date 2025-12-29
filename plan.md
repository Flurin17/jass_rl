## Architektur-Entscheid (damit RL bei Jass praktikabel wird)

Jass ist **4-Spieler, Team, Stichspiel mit unvollständiger Information**. Für “einfach RL” brauchst du:

* eine **exakte Regel-Engine** (deterministisch, testbar),
* eine **Multi-Agent Turn-based API** (z. B. PettingZoo AEC), ([PettingZoo][1])
* **Action Masking** (weil in Jass ständig Züge illegal sind), ([PettingZoo][2])
* und fürs Training ein Setup, das **Self-Play** oder “gegen Baselines” kann.

Als RL-Default: **(Maskable) PPO** mit Action Masking. ([Stable Baselines3 Contrib][3])

---

## Arbeitspakete (so, dass Codex CLI jeweils ein Paket “bauen + Tests” erledigen kann)

### AP0 — Scope festlegen: “Schieber” als Basis + Vollregeln schrittweise

**Deliverables**

* `docs/rules_scope.md` mit klarer Reihenfolge:

  1. Schieber Grundspiel (Trumpf/Obeabe/Uneufe, Stichzwang/Farbzwang/Trumpfzwang, Punkte, letzter Stich +5)
  2. **Stöck** (Trumpf König + Trumpf Ober) 20 Punkte ([lebendige-traditionen.ch][4])
  3. **Weis** (Ansagen) nach Schweizer Reglement (separates Modul, ein-/ausschaltbar) ([lebendige-traditionen.ch][4])
  4. Varianten/Turnieroptionen (falls nötig)
* Referenzregeln/Definitionen (Quellen, die du im Repo verlinkst), z. B. Kartenwerte und Total 157 ohne Weis/Stöck. ([Pagat][5])

**Tests**

* Noch keine, nur Abnahmekriterien.

---

### AP1 — Karten-/Regelmodel (Domain Layer, ohne RL)

**Deliverables**

* `core/cards.py`: Farben, Ränge, Kartenrepräsentation (36 Karten)
* `core/scoring.py`: Kartenwerte + Stichwerte für Trumpf / Nebenfarbe / Obeabe / Uneufe (Total 157 inkl. letzter Stich +5) ([Pagat][5])
* `core/rankings.py`: Stich-Comparator (welche Karte sticht welche) je Modus

**Tests (Unit, exhaustiv)**

* Für jede Modusart: Ranking-Tabellen stimmen (z. B. Trumpf: Buur > Nell > Ass …) ([Swisslos][6])
* Summe der Kartenpunkte pro Runde = 152 + 5 letzter Stich = 157 (ohne Weis/Stöck) ([Pagat][5])

---

### AP2 — Legal-Move Engine (das kritischste Stück)

**Deliverables**

* `core/legal_moves.py`: Funktion `legal_cards(hand, trick, trump_mode, trump_suit, partner_status, ruleset)`

  * implementiert **Farbzwang / Trumpfzwang / Stechzwang** für Schieber-Regelset
  * als “Ruleset Flags” konfigurierbar (weil es regionale/turnierbedingte Abweichungen gibt)

**Tests (Property + Tabellen)**

* “Wenn Farbe vorhanden → nur diese Farbe legal”
* “Wenn nicht Farbe vorhanden, aber Trumpfzwang aktiv → Trumpf muss, sofern vorhanden”
* “Wenn Trumpf im Stich liegt → muss trumpfen/übertrumpfen gemäss Regel”
* Randomisierte Tests: für 10’000 zufällige Hände/Stiche ist `legal_cards` nie leer (sofern Hand nicht leer), und alle legalen Karten sind wirklich spielbar.

Quellenbezug für Grundprinzipien (Farb-/Trumpfzwang etc.) ([Swisslos][6])

---

### AP3 — Spielzustand & Game Loop (komplettes Jass ohne Ansagen)

**Deliverables**

* `core/state.py`: Deal, Turn order, Trick state (4 Karten), Stiche, Punkte Team A/B, Restkarten
* `core/game.py`: `play_round(policy_by_player)` → kompletter Rundendurchlauf (9 Stiche)

**Tests (Integration)**

* Determinismus: gleicher Seed + gleiche Policies ⇒ identische Trajektorie
* Nach 9 Stichen: jeder Spieler hat 0 Karten, Punkte sind konsistent, letzter Stich +5 vergeben ([Pagat][5])

---

### AP4 — Trumpf-/Schieber-Phase (Bidding)

**Deliverables**

* `core/bidding.py`: Schieberlogik:

  * Spieler 1: “wähle Modus” oder “schiebe”
  * wenn geschoben: Partner wählt Modus
  * Modus: {Trumpf Farbe, Obeabe, Uneufe}

**Tests**

* Schiebe-Flow korrekt (wer darf wann wählen)
* Ungültige Auswahl wird verhindert (Action Masking später)

---

### AP5 — Stöck & Weis (Vollregeln als optionale Module)

**Deliverables**

* `core/announcements/stock.py`: Stöck Erkennung + Zeitpunkt des Weisens + 20 Punkte ([lebendige-traditionen.ch][4])
* `core/announcements/weis.py`: Weis-Regeln gemäss Schweizer Jassreglement (Kombinationen, Priorität, “melden/kontern”, Punkte)
* `ruleset.py`: Konfig, um Weis/Stöck an/aus zu schalten (für RL-Start evtl. Weis aus, später an)

**Tests**

* Stöck: nur Trumpf (König+Ober) zählt, nur wenn korrekt “gewiesen” beim Ausspielen der zweiten Karte ([lebendige-traditionen.ch][4])
* Weis: deterministische Beispielcases (fixe Hände) ⇒ erwartete Punkte/Validität

---

### AP6 — CLI Spiel (Menschen spielbar, Debug freundlich)

**Deliverables**

* `cli/play.py`: 4 Spieler (Human/AI beliebig mischbar), Text-UI
* `cli/replay.py`: Replays speichern (JSON) und abspielen

**Tests**

* Smoke: ein kompletter Roundtrip läuft ohne Crash
* Replay: gespeichertes Replay reproduziert exakt

---

### AP7 — RL Environment (Multi-Agent, action-masked)

**Deliverables**

* `env/jass_aec_env.py`: PettingZoo **AEC** Environment (4 agents: p0..p3) ([PettingZoo][1])
* Observation (suggested):

  * eigene Hand (36-dim one-hot oder 9 Karten encoded)
  * aktuelle Stichkarten (bis 3 bereits gespielt)
  * Trumpfmodus/-farbe
  * bisherige Stiche/gespielte Karten (public info)
  * Punkte/Tricks Team
  * optional: “Belief features” später
* Action space: **36 diskrete Aktionen** (jede Karte), plus in Bidding-Phase separate Actions
* Action mask: legal nur die Karten aus `legal_moves` ([PettingZoo][2])

**Tests**

* In jedem Zustand sind maskierte Aktionen genau die legalen Karten
* PettingZoo API Compliance (step/reset/agent order)

---

### AP8 — RL Training (Self-play + Baselines)

**Deliverables**

* `rl/train_selfplay.py`:

  * Start: trainiere p0 gegen 3 regelbasierte Bots
  * Danach: iteratives Self-Play (Checkpoint-Pool, Gegner sampling)
* Algorithmus: **MaskablePPO** (SB3 Contrib) ([Stable Baselines3 Contrib][3])
* `rl/eval.py`:

  * ELO/Winrate vs Baselines
  * fixed seeds, reproduzierbare Benchmarks

**Tests**

* Smoke-Train: 1–2k steps laufen in CI, Modell speicherbar, eval läuft
* Masking-Test: Train loop ruft nie eine invalid action (assert)

Hinweis: PettingZoo hat ein SB3+MaskablePPO Tutorial (Connect Four) und weist auf Version-Kompatibilitäten hin; für Stabilität in deinem Repo Versionen pinnen. ([PettingZoo][7])

---

### AP9 — “Fertig” Definition (Abnahmekriterien)

**Game-fertig**

* Alle Regeln aus deinem Scope (inkl. Stöck/Weis) implementiert + Tests grün
* CLI spielbar (4 Humans, oder 1 Human + 3 Bots)
* Replays

**RL-fertig**

* Train/Eval/Play Commands
* Agent schlägt Baseline-Bots signifikant (z. B. >60% Winrate über 1’000 Runden, fixed seeds)
* Dokumentation: “Wie trainieren”, “Wie debuggen”

---

## Wie Codex CLI das effizient abarbeitet

Pro Arbeitspaket jeweils als “Codex-Auftrag” formulieren:

* “Implementiere `core/legal_moves.py` gemäss tests in `tests/test_legal_moves.py`”
* “Erzeuge property-based tests für 10’000 random states”
* “Refactor: extrahiere ranking tables, keine Duplikate”
  Codex CLI ist genau für “Task → Code + Tests → Run” geeignet. ([Schweizer Jassverzeichnis][8])

---

## Empfehlung zur Reihenfolge (damit du nicht in Weis versinkst)

1. AP1–AP4 komplett (Schieber ohne Weis/Stöck)
2. AP7–AP8 RL lauffähig (gegen Baselines)
3. AP5 Weis/Stöck hinzufügen + Tests
4. RL erneut trainieren

Wenn du “alle Regeln” wirklich strikt nach Schweizer Jassreglement willst: nimm das offizielle Reglement aus dem PDF als normative Referenz und schreibe daraus Testfälle (“spec tests”), bevor Codex implementiert. ([lebendige-traditionen.ch][4])

[1]: https://pettingzoo.farama.org/api/aec/?utm_source=chatgpt.com "AEC API"
[2]: https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/?utm_source=chatgpt.com "Tutorial: Action Masking"
[3]: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html?utm_source=chatgpt.com "Maskable PPO - Stable Baselines3 - Contrib - Read the Docs"
[4]: https://www.lebendige-traditionen.ch/dam/tradition/fr/dokumente/tradition/ch/jassen.pdf.download.pdf/jassen.pdf?utm_source=chatgpt.com "Jassen"
[5]: https://www.pagat.com/de/jass/swjass.html?utm_source=chatgpt.com "Schweizer Jass - Allgemeine Regeln"
[6]: https://www.swisslos.ch/de/jass/informationen/jass-regeln/jass-grundlagen.html?utm_source=chatgpt.com "Jass-Regeln | Jass-Grundlagen"
[7]: https://pettingzoo.farama.org/tutorials/sb3/connect_four/?utm_source=chatgpt.com "SB3: Action Masked PPO for Connect Four"
[8]: https://jassverzeichnis.ch/schieber/?utm_source=chatgpt.com "Schieber-Jass: Eine Schritt-für-Schritt Anleitung"
