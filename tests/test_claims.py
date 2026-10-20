"""Jede Zahl aus der README wird hier über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250, c = 6, wo nichts anderes steht; Spread jeder Regel auf 600 unabhängigen Schnappschüssen geteilt durch den Spread von
Greedy (300 Trainings-Schnappschüsse), Median über die acht Netze. Regeln mit Gleichständen: Mittel über zehn zufällige Auflösungen. Zufall läuft über seedbare Generatoren und ist
plattformstabil; Verhältnisse werden mit Band geprüft (Schätzrauschen etwa ein bis zwei Prozentpunkte)."""

from functools import lru_cache

import pytest

import ifm_algorithm as A
import ifm_constants as C
import ifm_evaluation as ev

SW = C.SWEEP_SEEDS
K = C.CURVE_KS
ORDER = (C.M_DEGREE, C.M_CLOSENESS, C.M_BETWEENNESS, C.M_PAGERANK, C.M_CORE, C.M_DD, C.M_SD, C.M_DD_INDEX, C.M_RANDOM)


@lru_cache(maxsize=None)
def _curves(strength):
    return {r["typ"]: r for r in ev.curve_rows(250, 6, strength, seeds=SW)}


def _check(strength, typ, k, expect, tol=0.035):
    row = _curves(strength)[typ]
    for m, value in zip(ORDER, expect):
        assert ev.ratio_at(row, m, k) == pytest.approx(value, abs=tol), (strength, typ, k, m)


def test_ratios_at_ten_starters_and_weak_spread():
    t = 3
    _check(t, C.T_RANDOM, 10, (0.97, 0.92, 0.96, 0.96, 0.83, 1.00, 1.00, 1.00, 0.78))
    _check(t, C.T_POWER, 10, (0.97, 0.96, 0.98, 0.97, 0.94, 1.00, 0.98, 1.00, 0.80))
    _check(t, C.T_BA, 10, (0.97, 0.96, 0.97, 0.97, 0.76, 1.00, 0.98, 1.00, 0.75))
    _check(t, C.T_COMMUNITIES, 10, (0.95, 0.88, 0.95, 0.96, 0.82, 1.00, 0.99, 1.00, 0.77))
    _check(t, C.T_RING, 10, (0.89, 0.67, 0.85, 0.92, 0.82, 0.94, 0.94, 0.93, 0.82))
    _check(t, C.T_GRID, 10, (0.87, 0.45, 0.47, 0.52, 0.83, 0.89, 0.89, 0.63, 0.82))
    _check(t, C.T_HUBS, 10, (0.99, 0.69, 0.99, 0.99, 0.69, 1.00, 0.99, 1.00, 0.69))


def test_ratios_at_ten_starters_and_strong_spread():
    t = 15
    _check(t, C.T_RANDOM, 10, (0.98, 0.96, 0.98, 0.98, 0.97, 1.00, 0.99, 1.00, 0.96))
    _check(t, C.T_POWER, 10, (0.92, 0.91, 0.92, 0.92, 0.92, 0.95, 0.92, 0.95, 0.96))
    _check(t, C.T_BA, 10, (0.94, 0.94, 0.94, 0.94, 0.97, 0.97, 0.94, 0.97, 0.97))
    _check(t, C.T_COMMUNITIES, 10, (0.96, 0.94, 0.97, 0.97, 0.95, 0.99, 0.98, 0.99, 0.94))
    _check(t, C.T_RING, 10, (0.90, 0.76, 0.89, 0.92, 0.91, 0.93, 0.93, 0.91, 0.91))
    _check(t, C.T_GRID, 10, (0.91, 0.60, 0.62, 0.71, 0.91, 0.93, 0.93, 0.70, 0.90))
    _check(t, C.T_HUBS, 10, (0.98, 0.71, 0.98, 0.98, 0.71, 0.98, 0.98, 0.98, 0.84))


def test_the_gap_depends_on_k_hubs_at_the_edge_and_the_kernel_rule():
    hubs = _curves(15)[C.T_HUBS]
    assert ev.ratio_at(hubs, C.M_DEGREE, 1) == pytest.approx(0.46, abs=0.05) and ev.ratio_at(hubs, C.M_DEGREE, 3) == pytest.approx(0.86, abs=0.05) and ev.ratio_at(hubs, C.M_DEGREE, 5) == pytest.approx(0.99, abs=0.04) and ev.ratio_at(hubs, C.M_DEGREE, 10) == pytest.approx(0.98, abs=0.04)
    assert ev.ratio_at(hubs, C.M_CORE, 1) == pytest.approx(1.0, abs=0.04) and ev.ratio_at(hubs, C.M_CORE, 20) == pytest.approx(0.68, abs=0.05)
    ba = _curves(3)[C.T_BA]
    assert ev.ratio_at(_curves(3)[C.T_POWER], C.M_CORE, 1) == pytest.approx(0.62, abs=0.06)
    assert ev.ratio_at(ba, C.M_CORE, 1) == pytest.approx(0.24, abs=0.06) and ev.ratio_at(ba, C.M_CORE, 20) == pytest.approx(0.87, abs=0.05)
    grid = _curves(3)[C.T_GRID]
    assert ev.ratio_at(grid, C.M_CLOSENESS, 1) == pytest.approx(1.0, abs=0.05) and ev.ratio_at(grid, C.M_CLOSENESS, 20) == pytest.approx(0.45, abs=0.05)


def test_degree_discount_with_the_index_rule_loses_on_regular_networks():
    for strength, (index_grid, random_grid) in ((3, (0.63, 0.89)), (15, (0.70, 0.93))):
        grid = _curves(strength)[C.T_GRID]
        assert ev.ratio_at(grid, C.M_DD_INDEX, 10) == pytest.approx(index_grid, abs=0.04) and ev.ratio_at(grid, C.M_DD, 10) == pytest.approx(random_grid, abs=0.04)
    ring = _curves(15)[C.T_RING]
    assert ev.ratio_at(ring, C.M_DD_INDEX, 10) == pytest.approx(0.91, abs=0.04) and ev.ratio_at(ring, C.M_DD, 10) == pytest.approx(0.93, abs=0.04)


def test_curves_verdict_numbers():
    kind, text = ev.verdict_curves(list(_curves(3).values()), 10)
    assert "in 5 von 7 Netztypen mindestens 95 % von Greedy, am schwächsten in „Gitter“ mit 87 %" in text and "Die Nähe liegt in „Gitter“ nur bei 45 %" in text and "größter Unterschied in „Gitter“ (89 % gegen 63 %)" in text
    kind, text = ev.verdict_curves(list(_curves(15).values()), 10)
    assert "in 3 von 7 Netztypen mindestens 95 % von Greedy, am schwächsten in „Ring mit Abkürzungen“ mit 90 %" in text and "Die Nähe liegt in „Gitter“ nur bei 60 %" in text


@lru_cache(maxsize=None)
def _distance():
    return {r["typ"]: r for r in ev.distance_rows(250, 6, 15, 10, seeds=SW)}


def test_distance_and_redundancy_of_the_starters():
    rows = _distance()
    expect = {C.T_RANDOM: ((3.11, 0.24), (2.44, 0.20), (2.20, 0.19)), C.T_POWER: ((3.72, 0.38), (1.38, 0.11), (1.32, 0.11)), C.T_BA: ((2.99, 0.24), (1.57, 0.13), (1.50, 0.13)),
              C.T_COMMUNITIES: ((3.30, 0.26), (2.57, 0.20), (2.39, 0.21)), C.T_RING: ((5.60, 0.38), (3.73, 0.27), (2.47, 0.22)), C.T_GRID: ((9.94, 0.39), (7.80, 0.31), (1.73, 0.17)),
              C.T_HUBS: ((3.70, 0.35), (2.36, 0.19), (1.31, 0.10))}
    for typ, cells in expect.items():
        for m, (dist, red) in zip((C.M_GREEDY, C.M_DEGREE, C.M_CLOSENESS), cells):
            assert rows[typ][m]["dist"] == pytest.approx(dist, abs=0.25) and rows[typ][m]["red"] == pytest.approx(red, abs=0.03), (typ, m)
    kind, text = ev.verdict_distance(list(rows.values()))
    assert "„Potenzgesetz-Netz“ weiter verteilte Starter: mittlerer Abstand 3,7 gegen 1,4 nach Grad, Redundanz 0,38 gegen 0,11" in text and "Die Nähe wählt in „Hubs am Rand“ Starter im Abstand 1,3, Redundanz 0,10" in text


def test_celf_saves_evaluations_with_the_same_selection():
    rows = {r["typ"]: r for r in ev.celf_rows(250, 6, 10, seeds=SW)}
    expect = {C.T_RANDOM: (1149, 1486), C.T_POWER: (593, 687), C.T_BA: (826, 1014), C.T_COMMUNITIES: (1110, 1410), C.T_RING: (1070, 1368), C.T_GRID: (992, 1346), C.T_HUBS: (638, 648)}
    for typ, (c10, c20) in expect.items():
        assert rows[typ]["cells"][10]["celf"] == pytest.approx(c10, abs=60) and rows[typ]["cells"][20]["celf"] == pytest.approx(c20, abs=70), typ
        assert rows[typ]["cells"][10]["same"] and rows[typ]["cells"][20]["same"]
    assert rows[C.T_RANDOM]["cells"][10]["naive"] == 2455 and rows[C.T_RANDOM]["cells"][20]["naive"] == 4810 and rows[C.T_GRID]["cells"][10]["naive"] == 2515        # n + (n - 1) + ...: 250 · 10 - 45 usw.
    kind, text = ev.verdict_celf(list(rows.values()))
    assert kind == "success" and "CELF spart 53 % bis 87 %" in text and "am wenigsten in „Zufallsnetz“ bei k = 10: 1149 statt 2455" in text and "am meisten in „Hubs am Rand“ bei k = 20: 648 statt 4810" in text


def test_guarantee_against_the_optimum_on_small_networks():
    result = ev.guarantee_rows()
    s = ev.guarantee_summary(result)
    assert s["cases"] == 80 and s["min"] == pytest.approx(0.922, abs=0.005) and s["median"] == pytest.approx(0.9926, abs=0.003) and s["optimal"] == 29 and s["bound"] == pytest.approx(0.632, abs=0.001)
    kind, text = ev.verdict_guarantee(result)
    assert "im Median 99,3 % des Optimums, mindestens 92,2 %, und trifft das Optimum in 29 von 80 Fällen" in text
    # jeder Einzelfall liegt über der Garantie
    assert min(min(v) for v in result["ratios"].values()) > C.GUARANTEE


def test_greedy_is_not_unique():
    rows = ev.noise_rows(C.T_RANDOM, 250, 6, 10, 10, seeds=SW)
    by = {r["train"]: r for r in rows}
    assert by[50]["distinct"] == 8 and by[300]["distinct"] == 8 and by[50]["spread"] == pytest.approx(98.8, abs=2.0) and by[300]["spread"] == pytest.approx(100.0, abs=2.0)
    assert by[50]["range"] == pytest.approx(2.5, abs=1.2) and by[300]["range"] == pytest.approx(1.8, abs=1.0)
    kind, text = ev.verdict_noise(rows)
    assert "liefern 8 Läufe im Median 8 verschiedene Starter-Mengen" in text and "mit 300: 8 verschiedene Mengen" in text


def test_default_network_numbers():
    """Standardnetz (Zufallsnetz, n = 150, c = 6, Seed 35, Ausbreitungsstärke 10 %, k = 5): die Zahlen der Voreinstellung."""
    an = ev.analyse(ev.Settings())
    assert (an.inst.m, an.k, an.naive_evals, an.celf_evals) == (450, 5, 740, 518) and an.prep.p == pytest.approx(0.191, abs=0.002)
    assert an.spread(C.M_GREEDY) == pytest.approx(52.1, abs=0.1) and an.se_greedy == pytest.approx(0.7, abs=0.1)
    assert an.ratio(C.M_DD) == pytest.approx(1.003, abs=0.002) and an.ratio(C.M_CORE) == pytest.approx(0.90, abs=0.005) and an.ratio(C.M_RANDOM) == pytest.approx(0.86, abs=0.005)
    assert an.dist["greedy"] == pytest.approx(2.4, abs=0.05) and an.redund["greedy"] == pytest.approx(0.45, abs=0.01)
    assert ev.verdict_main(an)[0] == "info"


def test_preset_networks():
    link = ev.analyse(ev.Settings(**C.PRESETS["Unser Link-Netz"]))
    assert link.prep.p == pytest.approx(0.085, abs=0.003) and link.spread(C.M_GREEDY) == pytest.approx(142.4, abs=0.5) and link.ratio(C.M_DEGREE) == pytest.approx(0.946, abs=0.005)
    names = link.inst.labels
    assert [names[x] for x in link.orders[C.M_GREEDY][0][:2]] == ["index.html", "404.html"]                      # Startseite zuerst, dann der einzige Knoten ohne Verlinkung (Gewinn 1)
    assert link.dist["greedy"] == pytest.approx(1.78, abs=0.05) and link.dist["method"] == pytest.approx(1.0, abs=0.01)
    grid = ev.analyse(ev.Settings(**C.PRESETS["Gitter: die Nähe versagt"]))
    assert grid.prep.p == pytest.approx(0.278, abs=0.003) and grid.spread(C.M_GREEDY) == pytest.approx(79.2, abs=0.5) and grid.ratio(C.M_CLOSENESS) == pytest.approx(0.582, abs=0.005)
    assert grid.dist["method"] == pytest.approx(1.73, abs=0.05) and grid.dist["greedy"] == pytest.approx(7.33, abs=0.1) and grid.redund["method"] == pytest.approx(0.22, abs=0.01)
    hubs = ev.analyse(ev.Settings(**C.PRESETS["Hubs am Rand"]))
    assert hubs.prep.p == pytest.approx(0.334, abs=0.003) and hubs.ratio(C.M_CORE) == pytest.approx(0.69, abs=0.01) and hubs.dist["method"] == pytest.approx(1.36, abs=0.05) and hubs.redund["method"] == pytest.approx(0.10, abs=0.01)
    ring = ev.analyse(ev.Settings(**C.PRESETS["Ring mit Abkürzungen"]))
    assert ring.ratio(C.M_DD_INDEX) == pytest.approx(0.906, abs=0.005) and ring.dist["method"] == pytest.approx(4.09, abs=0.05) and ring.dist["greedy"] == pytest.approx(4.69, abs=0.05)
    power = ev.analyse(ev.Settings(**C.PRESETS["Hubs und Abstand"]))
    assert power.ratio(C.M_DEGREE) == pytest.approx(0.901, abs=0.005) and power.dist["method"] == pytest.approx(1.6, abs=0.05) and power.dist["greedy"] == pytest.approx(3.44, abs=0.05)
    assert power.redund["method"] == pytest.approx(0.13, abs=0.01) and power.redund["greedy"] == pytest.approx(0.32, abs=0.01)


def test_estimator_noise_is_visible():
    """Verhältnisse über 1 sind Schätzrauschen: Greedy ist auf den Trainings-Schnappschüssen optimiert, nicht auf den Bewertungs-Schnappschüssen."""
    grid = _curves(3)[C.T_GRID]
    assert ev.ratio_at(grid, C.M_CLOSENESS, 1) > 0.97                                                              # die Nähe ist für einen einzelnen Starter gleichwertig bis marginal besser
    assert max(max(row["ratios"][m]) for row in _curves(15).values() for m in C.RULES) < 1.06                      # nie deutlich über 1
