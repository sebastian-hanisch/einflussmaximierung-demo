# Einflussmaximierung: wen informiert man zuerst? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-einflussmaximierung-demo.streamlit.app/)** -->

Fünftes Stück der **Netzwerkanalyse-Reihe** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, im Ausbreitungs-Arm der Auftakt; Zusammenfluss aus der [Demo zu Kaskaden](https://sebastianhanisch-kaskaden-demo.streamlit.app/) (Ausbreitung), der [Zentralitäts-Demo](https://sebastianhanisch-centrality-demo.streamlit.app/) (Grad, Nähe, Betweenness) und der [k-Kern-Demo](https://sebastianhanisch-k-core-demo.streamlit.app/) (Kernzahl, Stück 1). Anders als die Fall-Demos zeigt sie **ein** Problem an einem wachsenden Beispiel: einem Netz von Subunternehmern und Lieferanten (synthetisch; dazu das echte Link-Netz von sebastianhanisch.net), in dem eine Störmeldung oder ein Verfahrenswechsel möglichst viele Betriebe erreichen soll, obwohl nur *k* Betriebe direkt informiert werden.

Jeder Informierte gibt die Nachricht an jeden Partner mit derselben Wahrscheinlichkeit *p* weiter (**Independent Cascade**). Die Demo stellt **Greedy** (Kempe, Kleinberg, Tardos 2003, mit der Beschleunigung CELF) den einfachen Auswahlregeln gegenüber: Grad, Nähe, Betweenness, PageRank, Kernzahl, Degree Discount, Single Discount und Zufall.

## Kernfrage

**Wie viel verliert die einfache Regel gegenüber Greedy, und wo?** Der Gesamtplan vermutete: „die Gradregel liegt oft nahe bei Greedy“. Die Vormessung schärfte das auf den Hook, der in der Demo steht: der **Schritt von einem zu mehreren Startern**. Wer die Starter einzeln nach ihrer Wirkung auswählt, wählt Knoten mit überlappender Reichweite; Greedy zählt die Überlappung mit.

## Befunde und Korrekturen gegenüber dem Plan

- **Gradregel nahe bei Greedy: nur teilweise.** Bei k = 10 liegt sie in 5 von 7 Netztypen bei mindestens 95 % von Greedy (schwache Ausbreitung), bei starker Ausbreitung nur in 3 von 7; im Gitter, im Ring und im Potenzgesetz-Netz verliert sie bis zu 13 %. Das ist kleiner als vermutet, aber nicht überall klein.
- **Hook gewechselt:** Abstand und Redundanz der Starter statt des Vergleichs von Einzelreichweiten (siehe Tabelle).
- **Die Nähe ist die schlechteste Regel bei mehreren Startern**, obwohl sie in Stück 1 den Einzelstarter am besten vorhersagte: bei einem Starter ist sie gleichwertig, bei mehreren wählt sie benachbarte Knoten.
- **Die Kernzahl-Auswahl aus Stück 1 ist schwach** (bei kleinem k in heavy-tailed Netzen 0,24 bis 0,62 von Greedy); sie bleibt als einer von mehreren Kandidaten.
- **Gleichstandsregel als Fehlerquelle:** Degree Discount mit der Regel „kleinste Knotennummer“ verliert im Gitter 23 bis 26 Prozentpunkte gegenüber zufälliger Auflösung. Die Demo löst Gleichstände zufällig auf (Mittel über zehn Ziehungen) und zeigt die Index-Regel als eigenes Verfahren.
- **Greedy liefert keine eindeutige Menge:** acht Läufe auf frischen Schnappschüssen ergeben acht verschiedene Mengen mit fast gleichem Spread.
- **Mittlerer Abstand über verbundene Paare:** nicht erreichbare Paare (Link-Netz: der einzige Knoten ohne Verlinkung) zählen nicht mit.
- **Stand der Technik nur als Einordnung:** Reverse-Influence-Sampling (TIM+, IMM) rechnet heute große Netze; die Demo zeigt die Lehrfassung mit Schnappschüssen.

## Modell

Ungerichtete Netze, jede Kante gibt mit gleicher Wahrscheinlichkeit *p* weiter, jede Kante bekommt genau eine Chance. Das ist gleichbedeutend mit Erreichbarkeit im Netz, in dem jede Kante unabhängig mit Wahrscheinlichkeit *p* **bleibt** (Schnappschuss); der erwartete **Spread** einer Startmenge ist die mittlere Größe der Vereinigung der Zusammenhangskomponenten ihrer Knoten. Statt *p* stellt man die **Ausbreitungsstärke** ein, den Anteil der Knoten, den ein zufälliger **einzelner** Starter im Mittel erreicht; *p* wird per Bisektion berechnet. Greedy rechnet auf **300** festen Trainings-Schnappschüssen (NewGreedyIC, Chen, Wang, Yang 2009), jede Auswahl wird auf **600** davon unabhängigen bewertet. **CELF** liefert dieselbe Auswahl mit weniger Gewinn-Auswertungen. Formeln und Literatur stehen im 📐-Expander der App.

## Methodik

| Baustein | Inhalt |
|---|---|
| **Netze** | Generatoren aus `k-core-demo` (Stück 1) als Kopie: Zufallsnetz, Potenzgesetz-Netz, skalenfrei, Verbünde, Ring mit Abkürzungen, Gitter, Hubs am Rand; dazu das Link-Netz von sebastianhanisch.net (Schnappschuss vom 2026-09-30, alle Seiten) |
| **Regeln** | Grad, Nähe, Betweenness (Brandes), PageRank, Kernzahl, Degree Discount, Single Discount (jeweils zehn zufällige Auflösungen der Gleichstände), Degree Discount mit „kleinster Index“, Zufall (30 Ziehungen) |
| **Experiment 1** | Spread geteilt durch den Spread von Greedy über k und Netztypen |
| **Experiment 2** | mittlerer Abstand der Starter und Redundanz (Spread der Menge geteilt durch die Summe der Einzelspreads) für Greedy, Grad und Nähe |
| **Experiment 3** | Gewinn-Auswertungen von CELF gegen naives Greedy |
| **Experiment 4** | Garantie gegen Wirklichkeit: 40 kleine Zufallsnetze mit exaktem Spread (Aufzählung aller Kantenzustände), Greedy gegen das Optimum durch Aufzählung |
| **Experiment 5** | Greedy auf frischen Schnappschüssen wiederholen: wie viele verschiedene Mengen, wie viel Streuung des Spreads |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250, c = 6, wo nichts anderes steht; Spread geteilt durch den Spread von Greedy, Median; Verhältnisse knapp über 1 sind Schätzrauschen (Greedy ist auf den Trainings-Schnappschüssen optimiert, nicht auf den Bewertungs-Schnappschüssen).

| Frage | Befund |
|---|---|
| **Stimmen die Bausteine?** | Zusammenhangskomponenten der Schnappschüsse stimmen mit `networkx` überein; der Schnappschuss-Spread mit dem **exakten Spread** durch Aufzählung aller 2^m Kantenzustände (30 kleine Netze, Band 0,2 Knoten); CELF liefert dieselbe Auswahl wie naives Greedy (40 Netze); Nähe, Betweenness, PageRank und Kernzahl stimmen mit `networkx` überein (60 Netze); Degree Discount mit einer unabhängig geschriebenen Neuberechnung der Formel (`tests/test_oracle_einflussmaximierung.py`) |
| **Wie nah liegt die Gradregel bei Greedy?** (k = 10, Ausbreitungsstärke 3 % / 15 %) | Zufallsnetz **0,97 / 0,98**, Potenzgesetz 0,97 / **0,92**, skalenfrei 0,97 / 0,94, Verbünde 0,95 / 0,96, Ring **0,89 / 0,90**, Gitter **0,87 / 0,91**, Hubs am Rand 0,99 / 0,98. In 5 von 7 (schwache) bzw. 3 von 7 (starke Ausbreitung) Netztypen mindestens 0,95 |
| **Wie schlecht ist die Nähe?** (k = 10, 3 % / 15 %) | Gitter **0,45 / 0,60**, Ring 0,67 / 0,76, Hubs am Rand 0,69 / 0,71, Verbünde 0,88 / 0,94, Zufallsnetz 0,92 / 0,96. Bei **einem** Starter dagegen gleichwertig: Gitter bei k = 1 gleichauf mit Greedy, bei k = 20 nur 0,45 |
| **Und die anderen Maße?** (k = 10, 3 %) | Betweenness im Gitter 0,47, PageRank 0,52; Kernzahl 0,69 (Hubs, Zufall 0,83, skalenfrei 0,76); Zufall zwischen 0,69 und 0,82. **Degree Discount** (zufällige Auflösung der Gleichstände) erreicht in 5 von 7 Netztypen 1,00, im Gitter und im Ring 0,89 bis 0,94 |
| **Hubs am Rand: wie hängt der Verlust von k ab?** (15 %, Gradregel) | k = 1: **0,46**, k = 3: 0,86, k = 5: 0,99, k = 10: 0,98; die Kernzahl-Regel: k = 1: 1,00, k = 20: 0,68 (sie wählt zunächst die Kerne, später Randknoten) |
| **Wie viel verliert Degree Discount durch die Gleichstandsregel?** (k = 10) | Gitter: zufällige Auflösung 0,89 / 0,93 (3 % / 15 %), „kleinste Knotennummer“ **0,63 / 0,70**; Ring bei 15 %: 0,93 gegen 0,91 |
| **Wählt Greedy Starter weiter auseinander?** (k = 10, 15 %; mittlerer Abstand / Redundanz: Greedy gegen Grad gegen Nähe) | **Potenzgesetz-Netz 3,7 / 0,38 gegen 1,4 / 0,11 gegen 1,3 / 0,11**; skalenfrei 3,0 / 0,24 gegen 1,6 / 0,13; Zufallsnetz 3,1 / 0,24 gegen 2,4 / 0,20; Ring 5,6 gegen 3,7 gegen 2,5; Gitter 9,9 gegen 7,8 gegen **1,7** (Nähe, Redundanz 0,17); Hubs am Rand 3,7 / 0,35 gegen 2,4 / 0,19 gegen **1,3 / 0,10** |
| **Wie viel spart CELF?** (Ausbreitungsstärke 10 %, k = 10 und 20) | **53 % bis 87 %** der Gewinn-Auswertungen bei **gleicher** Auswahl in allen Netzen: Zufallsnetz 1 149 statt 2 455 (k = 10) und 1 486 statt 4 810 (k = 20), Potenzgesetz 593 und 687, Hubs am Rand 638 und **648 statt 4 810** |
| **Garantie gegen Wirklichkeit** (40 kleine Zufallsnetze, n = 7 bis 9, exakter Spread bei p = 0,5, k = 2 und 3) | Greedy erreicht im Median **99,3 %** des Optimums, mindestens **92,2 %**, und trifft das Optimum in **29 von 80** Fällen; die Garantie 1 − 1/e = 63,2 % ist weit entfernt |
| **Ist die Auswahl von Greedy eindeutig?** (Zufallsnetz, k = 10, acht Läufe auf frischen Schnappschüssen) | **Nein:** bei 50 wie bei 300 Trainings-Schnappschüssen **8 von 8 verschiedene Mengen**, aber der Spread streut um nur 2,5 bzw. 1,8 Knoten (Median 98,8 bzw. 100,0 von 250) |
| **Standardnetz** (Zufallsnetz, n = 150, c = 6, Seed 35, Ausbreitungsstärke 10 %, k = 5) | 450 Kanten, p = 0,191; Greedy 52,1 Knoten (Standardfehler 0,7); Degree Discount 100 %, Kernzahl 90 %, Zufall 86 %; mittlerer Abstand der Starter 2,4, Redundanz 0,45; Gewinn-Auswertungen 740 (naiv) gegen 518 (CELF) |
| **Preset Potenzgesetz-Netz** (n = 150, 15 %, k = 10, Grad) | Grad 90 % von Greedy; Abstand der Starter **1,6 gegen 3,4**, Redundanz **0,13 gegen 0,32** |
| **Preset Gitter** (10 %, k = 10, Nähe) | Nähe 58 % von Greedy; Abstand der Starter 1,7 gegen 7,3, Redundanz 0,22 gegen 0,52 |
| **Preset Hubs am Rand** (15 %, k = 10, Kernzahl) | Kernzahl 69 % von Greedy; Abstand 1,4 gegen 3,9, Redundanz 0,10 gegen 0,38 |
| **Link-Netz** (350 Seiten, 15 %, k = 10) | p = 0,085; Greedy 142,4 Knoten; Grad 94,6 %. Greedy wählt als Erstes die **Startseite**, als Zweites **404.html** (der einzige Knoten ohne Verlinkung, Gewinn 1: danach ist jeder andere Zuwachs kleiner); Abstand der Grad-Starter 1,0 |

### Hypothesen des Einzelplans

| | Hypothese | Ergebnis |
|---|---|---|
| H1 | Greedy ist die beste Auswahl | ✅ alle Verhältnisse höchstens 1,06 (Schätzrauschen, bewertet auf unabhängigen Schnappschüssen) |
| H2 | Die Gradregel liegt nahe bei Greedy | ⚠ in 5 von 7 (3 %) bzw. 3 von 7 (15 %) Netztypen bei mindestens 0,95; Ring, Gitter und Potenzgesetz-Netz bei hoher Ausbreitung 0,87 bis 0,92 |
| H3 | Die Nähe ist auch bei mehreren Startern gut | ❌ schlechteste Regel in Gitter, Ring und bei Hubs am Rand: sie wählt benachbarte Knoten |
| H4 | Greedy verteilt die Starter weiter | ✅ Abstand 3,0 bis 9,9 gegen 1,4 bis 7,8 nach Grad; Redundanz 0,24 bis 0,39 gegen 0,11 bis 0,31 |
| H5 | Die Kernzahl ist als Auswahlregel schwach | ✅ bei k = 1 in heavy-tailed Netzen 0,24 (skalenfrei) bis 0,62 (Potenzgesetz) von Greedy (3 %), bei k = 10 0,69 bis 0,83 (3 %) |
| H6 | CELF liefert dieselbe Auswahl mit weniger Auswertungen | ✅ 53 % bis 87 % weniger |
| H7 | Die Garantie 1 − 1/e ist praktisch nicht relevant | ✅ Minimum 92,2 % des Optimums |
| H8 | Die Menge ist nicht eindeutig, ihr Spread stabil | ✅ 8 von 8 verschiedene Mengen, Streuung 1,8 bis 2,5 Knoten |
| H9 | Degree Discount liegt nahe bei Greedy | ✅ 1,00 in 5 von 7 Netztypen; im Gitter und im Ring 0,89 bis 0,94, mit „kleinster Index“ im Gitter nur 0,63 / 0,70 |
| H10 | Im Link-Netz reicht die Gradregel für kleine k | ✅ k = 10: 94,6 %; die Startseite erreicht fast alles |

## Ehrliche Grenzen

- **Ungerichtet, gleiche Weitergabewahrscheinlichkeit, nur Independent Cascade.** Mit Gewichten, Richtungen, Schwellenmodellen oder Kosten je Betrieb sind Spread und Garantie andere Größen.
- **Kalibrierung über einen Einzelstarter.** Eine andere Kalibrierung ergibt andere Lücken; die Ausbreitungsstärke ist ein Regler, die Befunde gelten für 3 %, 10 % und 15 %.
- **Schätzung mit Schnappschüssen:** Standardfehler etwa ein bis zwei Knoten, Verhältnisse knapp über 1 sind Rauschen.
- **Synthetische Netze** bei n = 250 (live n ≤ 250), Mediane über acht feste Netze; die Streuung der Verhältnisse zwischen den Netzen ist nicht ausgewiesen. Das Link-Netz ist ein Sonderfall (die Startseite ist mit allen Seiten verlinkt).
- **Garantie-Experiment nur auf winzigen Netzen** (n = 7 bis 9), weil der exakte Spread eine Aufzählung braucht.
- **Literatur:** die bibliografischen Angaben im 📐-Expander sind bereits in früheren Stücken der Reihe nachgeschlagen (Kempe u. a., Leskovec u. a., Kitsak u. a., Tang u. a.); die Formel von Degree Discount stammt aus dem Originaltext von Chen, Wang, Yang (2009), die Arbeiten selbst sind nicht vollständig gelesen.

## Verifikation

- **Kern:** Handwerte auf Mini-Instanzen (`tests/conftest.py`, `tests/test_algorithm.py`): exakter Spread auf dem Pfad (2 und 1,75), Stern bei p = 1, zwei Dreiecke (Redundanz 1 und 1/2), Greedy und CELF mit Gleichständen, Degree Discount und Single Discount von Hand, Kernzahl, Abstände, Kalibrierung.
- **Orakel:** `tests/test_oracle_einflussmaximierung.py` – Komponenten gegen `networkx`, Schnappschuss-Schätzer gegen exakten Spread, Greedy gegen das Optimum durch Aufzählung (Garantie 1 − 1/e), CELF gegen Greedy, Monotonie und Submodularität, Maße gegen `networkx`, Degree Discount gegen die neu gerechnete Formel, Aufzählung aller Gleichstands-Reihenfolgen.
- **Netze und Auswahlen:** gültig in allen Ecken der Regler, deterministisch je Seed (auch über Prozesse mit verschiedenem Hash-Seed), Vorbereitung (Kalibrierung trifft die Ausbreitungsstärke), verschachtelte Präfixe der Auswahl.
- **Verdikte:** jede Schwelle und jede Zahl im Text an von Hand gesetzten Werten (`tests/test_verdicts.py`), Randfälle in `tests/test_edge_cases.py`.
- **Befunde:** jede Zahl dieser README in `tests/test_claims.py`.
- **Mutationsprüfung des Kerns** (`ifm_algorithm.py`, 143 Stellen, ohne die langsamen Aussagen-Tests in der Suite der Überlebenden): nach Ergänzung von Tests keine Überlebenden; 11 Stellen sind als äquivalent festgehalten (z. B. Schutzgrenzen von Schleifen, Kanten ohne Schleifen, Gleichheit zweier Fließkommazahlen bei stetigem Zufall, eine Iteration weniger der Bisektion), acht Mutanten laufen in Endlosschleifen oder zu lange (Zeitüberschreitung). Zustand in `tests/mutation_state.json`.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jeder Netztyp mit jedem Verfahren, Permalink, alle fünf Experimente) und Footer-Test.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/link_netz.json` | Link-Netz von sebastianhanisch.net (Schnappschuss vom 2026-09-30) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx, scipy (nur für die Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `ifm_algorithm.py` | Kern: Schnappschüsse, Spread, Greedy, CELF, Auswahlregeln, Abstand und Redundanz, exakter Spread |
| `ifm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ifm_evaluation.py` | Vorbereitung, Analyse, Verdikt-Logik, Messreihen über feste Netze |
| `ifm_layout.py` | Fester Kopf (Stylesheet) |
| `ifm_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `ifm_scenario.py` | Netze: Generatoren (Kopie aus `k-core-demo`), Link-Netz, Anordnung |
| `ifm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanzen und Handwerte, Orakel, Netze, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Befunde |

## Bewusst nicht umgesetzt

- **Reverse-Influence-Sampling** (TIM+, IMM, OPIM): der Stand der Technik für große Netze, hier nur als Einordnung (möglicher Ausbau, gelb).
- **Linear-Threshold-Modell, Gewichte, Richtungen, Kosten je Betrieb:** eigene Fragen; die Schwellenmodelle kommen in Stück 6.
- **Lernbasierte Löser** (Graph-Netze für Einflussmaximierung): Güte gegen Greedy umstritten, hier nicht gemessen.
- **Mehr als 20 Starter und Netze über 250 Knoten live:** die Schnappschuss-Rechnung wächst linear in den Läufen, die Betweenness in n · m.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
