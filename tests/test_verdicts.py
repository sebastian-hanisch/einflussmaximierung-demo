"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses

import numpy as np
import pytest

import ifm_constants as C
import ifm_evaluation as ev

NAN = float("nan")


def _an(method=C.M_GREEDY, k=5, greedy=50.0, overrides=None, edges=True, se=0.7, dist=(2.4, 1.6), red=(0.45, 0.13)):
    base = ev.analyse(ev.Settings(C.T_RANDOM, 60, 4, 3, 10, k, method))
    spreads = {m: greedy for m in C.METHODS}
    spreads.update(overrides or {})
    curves = {m: np.full(k, float(v)) for m, v in spreads.items()}
    inst = base.inst if edges else dataclasses.replace(base.inst, adj=tuple(() for _ in base.inst.adj))
    return dataclasses.replace(base, inst=inst, prep=dataclasses.replace(base.prep, p=0.191), curves=curves, se_greedy=se, method=method, k=k,
                               dist={"greedy": dist[0], "method": dist[1]}, redund={"greedy": red[0], "method": red[1]})


def test_de_uses_a_decimal_comma_and_a_dash_for_undefined_numbers():
    assert ev.de(1.26, 1) == "1,3" and ev.de(0.9487, 3) == "0,949" and ev.de(3, 0) == "3" and ev.de(float("nan"), 2) == "–" and ev.de(None) == "–"


def test_main_verdict_for_greedy_names_best_worst_and_random_rules():
    over = {C.M_DEGREE: 48.0, C.M_CLOSENESS: 40.0, C.M_BETWEENNESS: 45.0, C.M_PAGERANK: 46.0, C.M_CORE: 30.0, C.M_DD: 49.0, C.M_SD: 47.0, C.M_DD_INDEX: 44.0, C.M_RANDOM: 25.0}
    kind, text = ev.verdict_main(_an(overrides=over))
    assert kind == "info"
    assert "Bei k = 5 Startern erreicht Greedy im Mittel 50,0 von 60 Knoten (Standardfehler der Schätzung 0,7), p = 0,191." in text
    assert "Die beste einfache Regel ist „Degree Discount“ mit 98 % von Greedy, die schwächste „Kernzahl“ mit 60 %, zufällige Starter erreichen 50 %." in text
    assert "Mittlerer Abstand der Starter: 2,4, Überlappung der Reichweiten (Redundanz): 0,45." in text


def test_main_verdict_thresholds_for_a_rule():
    def verdict(value, method=C.M_DEGREE):
        return ev.verdict_main(_an(method=method, greedy=100.0, overrides={method: value}))
    assert verdict(97.0)[0] == "success" and verdict(96.9)[0] == "info" and verdict(90.0)[0] == "info" and verdict(89.9)[0] == "warning"
    kind, text = verdict(58.0, C.M_CLOSENESS)
    assert kind == "warning" and "„Nähe“ erreicht 58,0 Knoten, das sind **58 %** von Greedy." in text
    assert "liegen im Mittel 1,6 Schritte auseinander (Greedy: 2,4), Redundanz 0,13 (Greedy: 0,45)" in text
    assert ev.verdict_main(_an(method=C.M_RANDOM, greedy=100.0, overrides={C.M_RANDOM: 30.0}))[0] == "info"                  # die Grundlinie wird nie bewertet


def test_main_verdict_without_edges():
    kind, text = ev.verdict_main(_an(edges=False))
    assert kind == "warning" and "keine Kanten" in text


def _row(typ, k_values=(1, 10), **ratios):
    r = {m: [1.0] * len(k_values) for m in C.METHODS}
    for name, vals in ratios.items():
        r[getattr(C, name)] = list(vals)
    return {"typ": typ, "ks": list(k_values), "ratios": r, "greedy": [10.0] * len(k_values), "n": 100.0, "p": 0.2}


def test_curves_verdict_names_near_weakest_and_the_tie_rule_gap():
    rows = [_row("A", M_DEGREE=(1, 0.98), M_CLOSENESS=(1, 0.9), M_DD=(1, 0.99), M_DD_INDEX=(1, 0.99)),
            _row("B", M_DEGREE=(1, 0.88), M_CLOSENESS=(1, 0.58), M_DD=(1, 0.94), M_DD_INDEX=(1, 0.73))]
    kind, text = ev.verdict_curves(rows, 10)
    assert kind == "info"
    assert "Bei k = 10 erreicht die Gradregel in 1 von 2 Netztypen mindestens 95 % von Greedy, am schwächsten in „B“ mit 88 %." in text
    assert "Die Nähe liegt in „B“ nur bei 58 %." in text and "größter Unterschied in „B“ (94 % gegen 73 %)" in text
    assert "in 2 von 2 Netztypen" in ev.verdict_curves([_row("A", M_DEGREE=(1, 0.95)), _row("B", M_DEGREE=(1, 0.99))], 10)[1]            # genau 95 % zählt
    assert ev.verdict_curves(rows, 7)[0] == "warning" and ev.verdict_curves([], 10)[0] == "warning"
    assert ev.ratio_at(rows[1], C.M_CLOSENESS, 10) == 0.58 and ev.ratio_at(rows[1], C.M_CLOSENESS, 1) == 1
    with pytest.raises(ValueError):
        ev.ratio_at(rows[0], C.M_DEGREE, 5)


def _dist_row(typ, greedy, degree, closeness):
    return {"typ": typ, C.M_GREEDY: {"dist": greedy[0], "red": greedy[1]}, C.M_DEGREE: {"dist": degree[0], "red": degree[1]}, C.M_CLOSENESS: {"dist": closeness[0], "red": closeness[1]}}


def test_distance_verdict_names_the_largest_gap_and_the_closest_pair_choice():
    rows = [_dist_row("A", (2.8, 0.31), (2.2, 0.25), (2.1, 0.25)), _dist_row("B", (2.96, 0.32), (1.54, 0.16), (1.48, 0.16)), _dist_row("C", (3.4, 0.39), (2.3, 0.22), (1.3, 0.11))]
    kind, text = ev.verdict_distance(rows)
    assert kind == "info"
    assert "Am deutlichsten wählt Greedy in „B“ weiter verteilte Starter: mittlerer Abstand 3,0 gegen 1,5 nach Grad, Redundanz 0,32 gegen 0,16." in text
    assert "Die Nähe wählt in „C“ Starter im Abstand 1,3, Redundanz 0,11." in text
    assert ev.verdict_distance([_dist_row("A", (NAN, NAN), (NAN, NAN), (NAN, NAN))])[0] == "warning"


def test_celf_verdict_names_savings_and_the_selection():
    rows = [{"typ": "A", "cells": {10: {"naive": 1455.0, "celf": 703.0, "same": True}, 20: {"naive": 2810.0, "celf": 882.0, "same": True}}},
            {"typ": "B", "cells": {10: {"naive": 1455.0, "celf": 440.0, "same": True}, 20: {"naive": 2810.0, "celf": 458.0, "same": True}}}]
    kind, text = ev.verdict_celf(rows)
    assert kind == "success"
    assert "CELF spart 52 % bis 84 % der Gewinn-Auswertungen (am wenigsten in „A“ bei k = 10: 703 statt 1455; am meisten in „B“ bei k = 20: 458 statt 2810)." in text
    assert "Die Auswahl ist in jedem Netz dieselbe wie beim naiven Greedy." in text
    rows[1]["cells"][10]["same"] = False
    kind, text = ev.verdict_celf(rows)
    assert kind == "warning" and "**Die Auswahl weicht in mindestens einem Netz ab.**" in text
    assert ev.verdict_celf([{"typ": "A", "cells": {10: {"naive": NAN, "celf": NAN, "same": True}}}])[0] == "warning"


def test_guarantee_verdict_and_summary():
    result = {"ratios": {2: [1.0, 0.95], 3: [0.99, 0.9]}, "p": 0.5, "nets": 2}
    s = ev.guarantee_summary(result)
    assert s["median"] == pytest.approx(0.97) and s["min"] == 0.9 and s["optimal"] == 1 and s["cases"] == 4
    kind, text = ev.verdict_guarantee(result)
    assert kind == "info" and "im Median 97,0 % des Optimums, mindestens 90,0 %, und trifft das Optimum in 1 von 4 Fällen; die Garantie liegt bei 63,2 %" in text and "bei p = 0,5" in text


def test_noise_verdict_compares_the_two_training_sizes():
    rows = [{"train": 50, "runs": 8, "distinct": 8.0, "spread": 64.9, "range": 2.1}, {"train": 300, "runs": 8, "distinct": 7.0, "spread": 66.2, "range": 0.8}]
    kind, text = ev.verdict_noise(rows)
    assert kind == "info"
    assert "Mit 50 Trainings-Schnappschüssen liefern 8 Läufe im Median 8 verschiedene Starter-Mengen, aber nur eine Streuung des Spreads von 2,1 Knoten (Median 64,9)" in text
    assert "mit 300: 7 verschiedene Mengen, Streuung 0,8 (Median 66,2)" in text
    assert ev.verdict_noise([])[0] == "warning"
