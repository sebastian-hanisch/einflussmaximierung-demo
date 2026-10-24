"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Netztypen -----------------------------------------------------------------------------------------------------------------------------------
T_RANDOM = "Zufallsnetz"
T_POWER = "Potenzgesetz-Netz"
T_BA = "Skalenfreies Netz (BA)"
T_COMMUNITIES = "Verbünde"
T_RING = "Ring mit Abkürzungen"
T_GRID = "Gitter"
T_HUBS = "Hubs am Rand"
T_LINK = "Link-Netz von sebastianhanisch.net"

GENERATED_TYPES = (T_RANDOM, T_POWER, T_BA, T_COMMUNITIES, T_RING, T_GRID, T_HUBS)   # n und c wirken
FIXED_TYPES = (T_LINK,)                                                             # echtes Netz, alle Seiten
ALL_TYPES = GENERATED_TYPES + FIXED_TYPES
BLOCKS = 6
NAV_COUNT, NAV_DROP = "mitzählen", "weglassen"                                      # nur das Link-Netz kennt die Navigationsseiten; die App zeigt es mit allen Seiten
NAV_PAGES = ("index.html", "kontakt.html")

C_MEANING = {
    T_RANDOM: "mittlerer Grad des Zufallsnetzes",
    T_POWER: "mittlerer Grad (Gradfolge mit Potenzgesetz-Exponent 2,5, Konfigurationsmodell)",
    T_BA: "zwei Kanten je neuem Knoten, also m = c / 2 gerundet (jeder Knoten hat mindestens m Nachbarn)",
    T_COMMUNITIES: "mittlerer Grad; 80 % der Kanten bleiben im eigenen Verbund (6 Verbünde)",
    T_RING: "Nachbarn auf dem Ring (c gerade gerundet), dazu Abkürzungen mit 5 % je Ringkante",
    T_GRID: "Nachbarschaft: unter 6 vier Nachbarn, 6 bis 7 mit einer Diagonale, ab 8 mit beiden Diagonalen",
    T_HUBS: "Kernkontakte je Randknoten (c / 3 gerundet, mindestens 2); der Rest sind Blätter an 5 Hubs",
}

N_MIN, N_MAX, DEFAULT_N = 60, 250, 150    # Slider-Maximum: Schnappschüsse O(Läufe · m), Betweenness O(n · m), Layout O(n^2 * 80)
C_MIN, C_MAX, DEFAULT_C = 3, 12, 6
SEED_MAX, DEFAULT_SEED = 999999, 35
STRENGTH_MIN, STRENGTH_MAX, DEFAULT_STRENGTH = 3, 30, 10     # Ausbreitungsstärke: Anteil der Knoten, den ein zufälliger Einzelstarter im Mittel erreicht, in Prozent
K_MIN, K_MAX, DEFAULT_K = 1, 20, 5
SWEEP_SEEDS = tuple(range(100000, 100008))         # feste Netze für alle Sweeps: nie ein einzelner Seed als Befund
LIVE_SEEDS = SWEEP_SEEDS[:4]                       # so viele feste Netze je Zelle rechnet die App live (die README-Zahlen nutzen alle acht)

# --- Verfahren -------------------------------------------------------------------------------------------------------------------------------------
M_GREEDY, M_DEGREE, M_CLOSENESS, M_BETWEENNESS, M_PAGERANK, M_CORE, M_DD, M_SD, M_DD_INDEX, M_RANDOM = (
    "Greedy", "Grad", "Nähe", "Betweenness", "PageRank", "Kernzahl", "Degree Discount", "Single Discount", "Degree Discount (kleinster Index)", "Zufall")
METHODS = (M_GREEDY, M_DEGREE, M_CLOSENESS, M_BETWEENNESS, M_PAGERANK, M_CORE, M_DD, M_SD, M_DD_INDEX, M_RANDOM)
DEFAULT_METHOD = M_GREEDY
RULES = tuple(m for m in METHODS if m != M_GREEDY)
METHOD_HELP = {
    M_GREEDY: "Wählt nacheinander den Knoten mit dem größten Zuwachs an erwarteter Reichweite (Kempe, Kleinberg, Tardos 2003), gerechnet auf festen Schnappschüssen.",
    M_DEGREE: "Die k Knoten mit den meisten Kontakten.",
    M_CLOSENESS: "Die k Knoten mit der größten Nähe (kleinster mittlerer Abstand zu allen).",
    M_BETWEENNESS: "Die k Knoten, über die die meisten kürzesten Wege laufen.",
    M_PAGERANK: "Die k Knoten mit dem größten PageRank (Dämpfung 0,85).",
    M_CORE: "Die k Knoten mit der größten Kernzahl (Stück 1 der Reihe).",
    M_DD: "Wie Grad, aber nach jeder Wahl wird der Grad der Nachbarn abgewertet (Chen, Wang, Yang 2009); Gleichstände zufällig aufgelöst.",
    M_SD: "Wie Degree Discount, aber jeder Nachbar eines gewählten Starters verliert genau einen Kontakt; Gleichstände zufällig aufgelöst.",
    M_DD_INDEX: "Degree Discount mit der Gleichstandsregel „kleinste Knotennummer“: auf regelmäßigen Netzen wählt sie Nachbarn und verliert stark.",
    M_RANDOM: "k zufällige Knoten: die Grundlinie.",
}
TIE_DRAWS = 10                                     # Regeln mit Gleichständen: Mittel über so viele zufällige Auflösungen
RANDOM_DRAWS = 30
TRAIN_RUNS, EVAL_RUNS = 300, 600                   # Schnappschüsse: Greedy rechnet auf festen Trainings-Schnappschüssen, bewertet wird auf unabhängigen
CALIB_RUNS, CALIB_ITERS = 40, 18                   # Kalibrierung von p: Bisektion mit so vielen Schnappschüssen je Schritt
CURVE_KS = (1, 2, 3, 5, 10, 15, 20)

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------------
SMALL_NETS = 40                                    # E4: kleine Zufallsnetze mit exaktem Spread
SMALL_N = (7, 9)
SMALL_M = (8, 12)
SMALL_K = (2, 3)
SMALL_P = 0.5
NOISE_RUNS = 8                                     # E5: Greedy-Läufe auf frischen Schnappschüssen
NOISE_TRAIN = (50, 300)
CELF_KS = (10, 20)
GUARANTEE = 1 - 1 / 2.718281828459045              # 1 - 1/e

# --- Darstellung ---------------------------------------------------------------------------------------------------------------------------------
COLOR_EDGE = "rgba(120,120,120,0.30)"
COLOR_SELECT = "#d62728"
COLOR_LINE = "#4c78a8"
COLOR_A = "#2F6B65"
COLOR_B = "#c4623b"
COLOR_C = "#6b5b95"
METHOD_COLORS = {M_GREEDY: "#2F6B65", M_DEGREE: "#4c78a8", M_CLOSENESS: "#c4623b", M_BETWEENNESS: "#b279a2", M_PAGERANK: "#72b7b2", M_CORE: "#9d755d", M_DD: "#e8a33d",
                 M_SD: "#f58518", M_DD_INDEX: "#d62728", M_RANDOM: "#bab0ac"}

PRESETS = {
    "Zufallsnetz (Voreinstellung)": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": DEFAULT_STRENGTH, "k": DEFAULT_K, "method": M_GREEDY},
    "Hubs und Abstand": {"netztyp": T_POWER, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": 15, "k": 10, "method": M_DEGREE},
    "Gitter: die Nähe versagt": {"netztyp": T_GRID, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": DEFAULT_STRENGTH, "k": 10, "method": M_CLOSENESS},
    "Ring mit Abkürzungen": {"netztyp": T_RING, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": DEFAULT_STRENGTH, "k": 10, "method": M_DD_INDEX},
    "Hubs am Rand": {"netztyp": T_HUBS, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": 15, "k": 10, "method": M_CORE},
    "Unser Link-Netz": {"netztyp": T_LINK, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "strength": 15, "k": 10, "method": M_DEGREE},
}
PRESET_HELP = {
    "Zufallsnetz (Voreinstellung)": "Zufallsnetz, kleine Ausbreitung, fünf Starter: Greedy gegen die einfachen Regeln.",
    "Hubs und Abstand": "Potenzgesetz-Netz, hohe Ausbreitung: die Starter nach Grad liegen dicht beieinander und überdecken sich.",
    "Gitter: die Nähe versagt": "Auf dem Gitter wählt die Nähe benachbarte Knoten in der Mitte: eine der schwächsten Regeln (die Betweenness liegt gleichauf).",
    "Ring mit Abkürzungen": "Degree Discount mit der Regel „kleinste Knotennummer“ wählt hier Nachbarn; mit zufälliger Auflösung der Gleichstände ist es fast so gut wie Greedy.",
    "Hubs am Rand": "Wenige Hubs mit Blättern am Rand eines dichten Kerns: die Kernzahl wählt Randknoten ohne Wirkung.",
    "Unser Link-Netz": "Das echte Link-Netz von sebastianhanisch.net (alle Seiten): Startseite und Kontaktseite sind mit allen Seiten verlinkt.",
}
DEFAULT_PRESET_NAME = "Zufallsnetz (Voreinstellung)"
