"""Kennzahlen, Sweeps und Verdikte der Demo „Einflussmaximierung“. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Protokoll: Independent Cascade mit gleicher Weitergabewahrscheinlichkeit p auf ungerichteten Netzen; p wird so kalibriert, dass ein zufälliger Einzelstarter im Mittel „Ausbreitungsstärke“ Prozent
der Knoten erreicht. Greedy rechnet auf TRAIN_RUNS festen Schnappschüssen, bewertet wird jede Regel auf EVAL_RUNS davon unabhängigen. Regeln mit Gleichständen werden über TIE_DRAWS zufällige
Auflösungen gemittelt (die Regel „kleinster Index“ steht eigens daneben). Jede Messreihe nimmt den Median über feste Netze (`LIVE_SEEDS` in der App, `SWEEP_SEEDS` für die README-Zahlen)."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import ifm_algorithm as A
import ifm_constants as C
from ifm_scenario import Instance, make_instance

NAN = float("nan")


def de(x, digits=1) -> str:
    """Dezimalzahl mit Komma (deutsche Texte); eine nicht definierte Zahl erscheint als Gedankenstrich."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


def _med(values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(median(vals)) if vals else NAN


# --- Vorbereitung eines Netzes -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Prepared:
    adj: tuple
    p: float
    comp_t: np.ndarray      # Trainings-Schnappschüsse (Greedy rechnet darauf)
    size_t: np.ndarray
    comp_e: np.ndarray      # unabhängige Bewertungs-Schnappschüsse
    size_e: np.ndarray


def prepare(adj, strength, seed, train=C.TRAIN_RUNS, evaluate=C.EVAL_RUNS) -> Prepared:
    """Kalibriert p auf die Ausbreitungsstärke (Prozent) und zieht die beiden Schnappschuss-Mengen."""
    n = len(adj)
    u, v = A.edge_arrays(adj)
    p = A.calibrate_p(n, u, v, strength / 100.0, seed)
    comp_t, size_t = A.snapshot_components(n, u, v, p, train, seed + 1000)
    comp_e, size_e = A.snapshot_components(n, u, v, p, evaluate, seed + 2000)
    return Prepared(adj, p, comp_t, size_t, comp_e, size_e)


def method_orders(prep: Prepared, kmax, seed, methods=C.METHODS) -> dict:
    """Je Verfahren eine Liste von Starter-Reihenfolgen der Länge kmax (Greedy und die Regel „kleinster Index“: eine; Regeln mit Gleichständen: TIE_DRAWS zufällige Auflösungen; Zufall: RANDOM_DRAWS).
    Die ersten k Einträge jeder Reihenfolge sind die Auswahl für k Starter."""
    adj = prep.adj
    n = len(adj)
    kmax = min(kmax, n)
    rng = np.random.default_rng(int(seed) + 31)
    out = {}
    values = {}
    for m in methods:
        if m == C.M_GREEDY:
            out[m] = [A.greedy_select(prep.comp_t, prep.size_t, kmax)[0]]
        elif m == C.M_DD_INDEX:
            out[m] = [A.degree_discount_order(adj, kmax, prep.p, A.tiebreak_vector(n))]
        elif m == C.M_RANDOM:
            out[m] = [rng.permutation(n)[:kmax].tolist() for _ in range(C.RANDOM_DRAWS)]
        elif m == C.M_DD:
            out[m] = [A.degree_discount_order(adj, kmax, prep.p, A.tiebreak_vector(n, rng)) for _ in range(C.TIE_DRAWS)]
        elif m == C.M_SD:
            out[m] = [A.single_discount_order(adj, kmax, A.tiebreak_vector(n, rng)) for _ in range(C.TIE_DRAWS)]
        else:
            if m not in values:
                values[m] = {C.M_DEGREE: lambda: [float(d) for d in A.degrees(adj)], C.M_CLOSENESS: lambda: A.closeness(adj), C.M_BETWEENNESS: lambda: A.betweenness(adj),
                             C.M_PAGERANK: lambda: A.pagerank(adj), C.M_CORE: lambda: [float(c) for c in A.core_numbers(adj)]}[m]()
            out[m] = [A.ranking_by(values[m], A.tiebreak_vector(n, rng))[:kmax] for _ in range(C.TIE_DRAWS)]
    return out


def method_curves(prep: Prepared, orders: dict) -> dict:
    """Erwarteter Spread der ersten 1 bis kmax Starter je Verfahren (Mittel über die Auflösungen), bewertet auf den unabhängigen Schnappschüssen: Array je Verfahren."""
    return {m: np.mean([A.prefix_spreads(prep.comp_e, prep.size_e, o) for o in lst], axis=0) for m, lst in orders.items()}


# --- Hauptanalyse -----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    netztyp: str = C.T_RANDOM
    n: int = C.DEFAULT_N
    c: int = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED
    strength: int = C.DEFAULT_STRENGTH
    k: int = C.DEFAULT_K
    method: str = C.DEFAULT_METHOD


@dataclass(frozen=True, eq=False)
class Analysis:
    inst: Instance
    prep: Prepared
    k: int                      # wirksame Zahl der Starter (höchstens n)
    method: str
    orders: dict                # Verfahren -> Liste von Reihenfolgen
    curves: dict                # Verfahren -> Spread bei 1 bis k Startern
    chosen: list                # die Starter des gewählten Verfahrens (erste Auflösung)
    se_greedy: float            # Standardfehler des Greedy-Spreads bei k
    reach: np.ndarray           # Erreichungswahrscheinlichkeit je Knoten für `chosen`
    dist: dict                  # mittlerer Abstand der Starter: Greedy und gewähltes Verfahren
    redund: dict                # Redundanz: Greedy und gewähltes Verfahren
    celf_evals: int
    naive_evals: int

    def spread(self, method) -> float:
        return float(self.curves[method][self.k - 1])

    def ratio(self, method) -> float:
        g = self.spread(C.M_GREEDY)
        return self.spread(method) / g if g > 0 else NAN


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.netztyp, settings.n, settings.c, settings.seed, layout=True)
    prep = prepare(inst.adj, settings.strength, settings.seed)
    k = min(settings.k, inst.n)
    orders = method_orders(prep, k, settings.seed)
    curves = method_curves(prep, orders)
    chosen = list(orders[settings.method][0][:k])
    greedy = list(orders[C.M_GREEDY][0][:k])
    _, naive = A.greedy_select(prep.comp_t, prep.size_t, k)
    _, lazy = A.celf_select(prep.comp_t, prep.size_t, k)
    return Analysis(inst, prep, k, settings.method, orders, curves, chosen, A.spread_se(prep.comp_e, prep.size_e, greedy), A.reach_probability(prep.comp_e, chosen),
                    {"greedy": A.mean_pairwise_distance(inst.adj, greedy), "method": A.mean_pairwise_distance(inst.adj, chosen)},
                    {"greedy": A.redundancy(prep.comp_e, prep.size_e, greedy), "method": A.redundancy(prep.comp_e, prep.size_e, chosen)}, lazy, naive)


def verdict_main(an: Analysis) -> tuple[str, str]:
    """Verdikt der Kernfrage: wie viel verliert die einfache Regel gegenüber Greedy?"""
    n = an.inst.n
    greedy = an.spread(C.M_GREEDY)
    head = (f"Bei k = {an.k} Startern erreicht Greedy im Mittel {de(greedy, 1)} von {n} Knoten (Standardfehler der Schätzung {de(an.se_greedy, 1)}), p = {de(an.prep.p, 3)}.")
    if an.inst.m == 0:
        return "warning", "Das Netz hat keine Kanten: jeder Starter erreicht nur sich selbst."
    rules = [(m, an.ratio(m)) for m in C.RULES if m != C.M_RANDOM]
    best = max(rules, key=lambda x: x[1])
    worst = min(rules, key=lambda x: x[1])
    rnd = an.ratio(C.M_RANDOM)
    if an.method == C.M_GREEDY:
        return "info", (head + f" Die beste einfache Regel ist „{best[0]}“ mit {de(100 * best[1], 0)} % von Greedy, die schwächste „{worst[0]}“ mit {de(100 * worst[1], 0)} %, "
                        f"zufällige Starter erreichen {de(100 * rnd, 0)} %. Mittlerer Abstand der Starter: {de(an.dist['greedy'], 1)}, Überlappung der Reichweiten (Redundanz): {de(an.redund['greedy'], 2)}.")
    r = an.ratio(an.method)
    detail = (f" Die Starter von „{an.method}“ liegen im Mittel {de(an.dist['method'], 1)} Schritte auseinander (Greedy: {de(an.dist['greedy'], 1)}), Redundanz {de(an.redund['method'], 2)} (Greedy: {de(an.redund['greedy'], 2)}).")
    text = f"{head} „{an.method}“ erreicht {de(an.spread(an.method), 1)} Knoten, das sind **{de(100 * r, 0)} %** von Greedy." + detail
    if an.method == C.M_RANDOM:
        return "info", text
    if r >= 0.97:
        return "success", text
    if r >= 0.9:
        return "info", text
    return "warning", text


# --- E1: Spread über k und Netztypen --------------------------------------------------------------------------------------------------------------------


def curve_rows(n, c, strength, ks=C.CURVE_KS, seeds=C.LIVE_SEEDS, types=None) -> list[dict]:
    """Je Netztyp und Verfahren das Verhältnis von Spread und Greedy-Spread bei k = ks (Median über die festen Netze) und der Greedy-Spread selbst."""
    rows = []
    kmax = max(ks)
    for typ in types or C.GENERATED_TYPES:
        per_seed = []
        for seed in seeds:
            inst = make_instance(typ, n, c, seed, layout=False)
            prep = prepare(inst.adj, strength, seed)
            curves = method_curves(prep, method_orders(prep, kmax, seed))
            kk = [min(k, inst.n) for k in ks]
            per_seed.append({"curves": {m: [float(curves[m][k - 1]) for k in kk] for m in curves}, "n": inst.n, "p": prep.p})
        ratios = {m: [_med([ps["curves"][m][i] / ps["curves"][C.M_GREEDY][i] for ps in per_seed]) for i in range(len(ks))] for m in C.METHODS}
        rows.append({"typ": typ, "ks": list(ks), "ratios": ratios, "greedy": [_med([ps["curves"][C.M_GREEDY][i] for ps in per_seed]) for i in range(len(ks))],
                     "n": _med([float(ps["n"]) for ps in per_seed]), "p": _med([ps["p"] for ps in per_seed])})
    return rows


def ratio_at(row, method, k) -> float:
    return row["ratios"][method][row["ks"].index(k)]


def verdict_curves(rows: list[dict], k: int) -> tuple[str, str]:
    """Verdikt zu E1: wo liegen die einfachen Regeln nahe bei Greedy, wo nicht?"""
    if not rows or k not in rows[0]["ks"]:
        return "warning", "Für dieses k liegt keine Messreihe vor."
    deg = [(r["typ"], ratio_at(r, C.M_DEGREE, k)) for r in rows]
    clo = [(r["typ"], ratio_at(r, C.M_CLOSENESS, k)) for r in rows]
    near = [t for t, x in deg if x >= 0.95]
    wd = min(deg, key=lambda x: x[1])
    wc = min(clo, key=lambda x: x[1])
    dd = [(r["typ"], ratio_at(r, C.M_DD, k)) for r in rows]
    di = [(r["typ"], ratio_at(r, C.M_DD_INDEX, k)) for r in rows]
    gap = max(((t, a - b) for (t, a), (_, b) in zip(dd, di)), key=lambda x: x[1])
    return "info", (f"Bei k = {k} erreicht die Gradregel in {len(near)} von {len(rows)} Netztypen mindestens 95 % von Greedy, am schwächsten in „{wd[0]}“ mit {de(100 * wd[1], 0)} %. "
                    f"Die Nähe liegt in „{wc[0]}“ nur bei {de(100 * wc[1], 0)} %. Degree Discount mit zufälliger Auflösung der Gleichstände gegen die Regel „kleinster Index“: "
                    f"größter Unterschied in „{gap[0]}“ ({de(100 * dict(dd)[gap[0]], 0)} % gegen {de(100 * dict(di)[gap[0]], 0)} %).")


# --- E2: Abstand und Redundanz ----------------------------------------------------------------------------------------------------------------------------


def distance_rows(n, c, strength, k, seeds=C.LIVE_SEEDS, types=None) -> list[dict]:
    """Mittlerer Abstand der Starter und Redundanz (Spread geteilt durch die Summe der Einzelspreads) für Greedy, Grad und Nähe je Netztyp; Median über die festen Netze (Regeln: erste Auflösung)."""
    rows = []
    for typ in types or C.GENERATED_TYPES:
        acc = {m: {"dist": [], "red": []} for m in (C.M_GREEDY, C.M_DEGREE, C.M_CLOSENESS)}
        for seed in seeds:
            inst = make_instance(typ, n, c, seed, layout=False)
            prep = prepare(inst.adj, strength, seed)
            kk = min(k, inst.n)
            orders = method_orders(prep, kk, seed, methods=(C.M_GREEDY, C.M_DEGREE, C.M_CLOSENESS))
            for m in acc:
                sel = orders[m][0][:kk]
                acc[m]["dist"].append(A.mean_pairwise_distance(inst.adj, sel))
                acc[m]["red"].append(A.redundancy(prep.comp_e, prep.size_e, sel))
        rows.append({"typ": typ, **{m: {"dist": _med(acc[m]["dist"]), "red": _med(acc[m]["red"])} for m in acc}})
    return rows


def verdict_distance(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2: wählt Greedy Starter weiter auseinander als die einfachen Regeln?"""
    valid = [r for r in rows if not math.isnan(r[C.M_GREEDY]["dist"])]
    if not valid:
        return "warning", "Keine Abstände berechenbar."
    top = max(valid, key=lambda r: r[C.M_GREEDY]["dist"] - r[C.M_DEGREE]["dist"])
    clo = min(valid, key=lambda r: r[C.M_CLOSENESS]["dist"])
    return "info", (f"Am deutlichsten wählt Greedy in „{top['typ']}“ weiter verteilte Starter: mittlerer Abstand {de(top[C.M_GREEDY]['dist'], 1)} gegen {de(top[C.M_DEGREE]['dist'], 1)} nach Grad, "
                    f"Redundanz {de(top[C.M_GREEDY]['red'], 2)} gegen {de(top[C.M_DEGREE]['red'], 2)}. Die Nähe wählt in „{clo['typ']}“ Starter im Abstand {de(clo[C.M_CLOSENESS]['dist'], 1)}, Redundanz {de(clo[C.M_CLOSENESS]['red'], 2)}.")


# --- E3: CELF gegen naives Greedy -----------------------------------------------------------------------------------------------------------------------


def celf_rows(n, c, strength, ks=C.CELF_KS, seeds=C.LIVE_SEEDS, types=None) -> list[dict]:
    """Zahl der Gewinn-Auswertungen von CELF gegen naives Greedy (n + (n - 1) + ... je Runde) je Netztyp und k; `same` = alle festen Netze liefern dieselbe Auswahl; Median über die festen Netze."""
    rows = []
    for typ in types or C.GENERATED_TYPES:
        cells = {k: {"naive": [], "celf": [], "same": True} for k in ks}
        for seed in seeds:
            inst = make_instance(typ, n, c, seed, layout=False)
            prep = prepare(inst.adj, strength, seed, evaluate=2)
            for k in ks:
                g, naive = A.greedy_select(prep.comp_t, prep.size_t, min(k, inst.n))
                lazy_sel, lazy = A.celf_select(prep.comp_t, prep.size_t, min(k, inst.n))
                cells[k]["naive"].append(float(naive))
                cells[k]["celf"].append(float(lazy))
                cells[k]["same"] = cells[k]["same"] and g == lazy_sel
        rows.append({"typ": typ, "cells": {k: {"naive": _med(v["naive"]), "celf": _med(v["celf"]), "same": v["same"]} for k, v in cells.items()}})
    return rows


def verdict_celf(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E3: wie viele Auswertungen spart CELF, und bleibt die Auswahl gleich?"""
    cells = [(r["typ"], k, v) for r in rows for k, v in r["cells"].items() if not math.isnan(v["naive"])]
    if not cells:
        return "warning", "Keine Zelle berechenbar."
    saving = [(1 - v["celf"] / v["naive"], t, k, v) for t, k, v in cells]
    lo, hi = min(saving, key=lambda x: x[0]), max(saving, key=lambda x: x[0])
    same = all(v["same"] for _, _, v in cells)
    return ("success" if same else "warning"), (f"CELF spart {de(100 * lo[0], 0)} % bis {de(100 * hi[0], 0)} % der Gewinn-Auswertungen (am wenigsten in „{lo[1]}“ bei k = {lo[2]}: {de(lo[3]['celf'], 0)} statt {de(lo[3]['naive'], 0)}; "
                                                f"am meisten in „{hi[1]}“ bei k = {hi[2]}: {de(hi[3]['celf'], 0)} statt {de(hi[3]['naive'], 0)}). " + ("Die Auswahl ist in jedem Netz dieselbe wie beim naiven Greedy." if same else "**Die Auswahl weicht in mindestens einem Netz ab.**"))


# --- E4: Garantie gegen Wirklichkeit ----------------------------------------------------------------------------------------------------------------------


def small_network(seed):
    """Kleines Zufallsnetz für den exakten Spread: n zwischen 7 und 9, m zwischen 8 und 12 verschiedenen Kanten. Rückgabe (n, Kantenliste)."""
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(C.SMALL_N[0], C.SMALL_N[1] + 1))
    m = int(rng.integers(C.SMALL_M[0], C.SMALL_M[1] + 1))
    pairs = set()
    while len(pairs) < m:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        pairs.add((a, b))
    return n, sorted(pairs)


def guarantee_rows(nets=C.SMALL_NETS, ks=C.SMALL_K, p=C.SMALL_P, first_seed=100000) -> dict:
    """Greedy (mit dem exakten Spread) gegen das Optimum durch Aufzählung aller k-Mengen auf kleinen Zufallsnetzen. Rückgabe je k die Verhältnisse Greedy / Optimum aller Netze."""
    ratios = {k: [] for k in ks}
    for i in range(nets):
        n, edges = small_network(first_seed + i)
        table = A.exact_table(n, edges, p)
        for k in ks:
            greedy = A.exact_greedy(table, n, k)
            _, best = A.exact_optimum(table, n, k)
            ratios[k].append(A.exact_spread(table, greedy) / best)
    return {"ratios": ratios, "nets": nets, "p": p}


def guarantee_summary(result: dict) -> dict:
    allr = [r for v in result["ratios"].values() for r in v]
    return {"median": float(median(allr)), "min": float(min(allr)), "optimal": sum(1 for r in allr if r > 1 - 1e-9), "cases": len(allr), "bound": C.GUARANTEE}


def verdict_guarantee(result: dict) -> tuple[str, str]:
    """Verdikt zu E4: wie weit liegt Greedy tatsächlich vom Optimum, gegen die Garantie 1 − 1/e?"""
    s = guarantee_summary(result)
    return "info", (f"Greedy erreicht im Median {de(100 * s['median'], 1)} % des Optimums, mindestens {de(100 * s['min'], 1)} %, und trifft das Optimum in {s['optimal']} von {s['cases']} Fällen; "
                    f"die Garantie liegt bei {de(100 * s['bound'], 1)} % und liegt weit darunter (nur gemessen auf kleinen Netzen mit n = 7 bis 9 und exaktem Spread bei p = {de(result['p'], 1)}).")


# --- E5: Greedy ist nicht eindeutig -------------------------------------------------------------------------------------------------------------------------


def noise_rows(typ, n, c, strength, k, seeds=C.LIVE_SEEDS, runs=C.NOISE_RUNS, trains=C.NOISE_TRAIN) -> list[dict]:
    """Greedy-Läufe auf je frischen Trainings-Schnappschüssen (50 und 300): Zahl verschiedener Starter-Mengen, Streuung des unabhängig bewerteten Spreads; Median über die festen Netze."""
    out = []
    for train in trains:
        distinct, med_spread, spread_range = [], [], []
        for seed in seeds:
            inst = make_instance(typ, n, c, seed, layout=False)
            prep = prepare(inst.adj, strength, seed, train=train)
            kk = min(k, inst.n)
            u, v = A.edge_arrays(inst.adj)
            sets, spreads = [], []
            for r in range(runs):
                comp, size = A.snapshot_components(inst.n, u, v, prep.p, train, seed + 5000 + 17 * r)
                sel = A.greedy_select(comp, size, kk)[0]
                sets.append(frozenset(sel))
                spreads.append(A.spread_of(prep.comp_e, prep.size_e, sel))
            distinct.append(float(len(set(sets))))
            med_spread.append(float(median(spreads)))
            spread_range.append(float(max(spreads) - min(spreads)))
        out.append({"train": train, "runs": runs, "distinct": _med(distinct), "spread": _med(med_spread), "range": _med(spread_range)})
    return out


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E5: viele verschiedene Mengen, aber fast derselbe Spread?"""
    if not rows:
        return "warning", "Keine Läufe."
    a, b = rows[0], rows[-1]
    return "info", (f"Mit {a['train']} Trainings-Schnappschüssen liefern {de(a['runs'], 0)} Läufe im Median {de(a['distinct'], 0)} verschiedene Starter-Mengen, aber nur eine Streuung des Spreads von {de(a['range'], 1)} Knoten (Median {de(a['spread'], 1)}); "
                    f"mit {b['train']}: {de(b['distinct'], 0)} verschiedene Mengen, Streuung {de(b['range'], 1)} (Median {de(b['spread'], 1)}). Es gibt viele fast gleich gute Auswahlen: die konkrete Menge ist nicht eindeutig.")
