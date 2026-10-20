"""Netze: gültige Netze, Determinismus (auch über Prozesse), Positionen, Link-Netz, Hashseed-Unabhängigkeit; Vorbereitung (Kalibrierung, Schnappschüsse) und Verfahren."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ifm_algorithm as A
import ifm_constants as C
import ifm_evaluation as ev
import ifm_scenario as S

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("typ", C.GENERATED_TYPES)
@pytest.mark.parametrize("n,c", [(C.N_MIN, C.C_MIN), (C.N_MIN, C.C_MAX), (C.DEFAULT_N, C.DEFAULT_C), (C.N_MAX, C.C_MAX)])
def test_every_corner_gives_a_valid_network(typ, n, c):
    inst = S.make_instance(typ, n, c, 35)
    assert inst.n >= 2 and len(inst.pos) == inst.n
    for x, y in inst.pos:
        assert math.isfinite(x) and math.isfinite(y) and -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
    for a, nb in enumerate(inst.adj):
        assert list(nb) == sorted(set(nb)) and a not in nb
        for b in nb:
            assert a in inst.adj[b]


def test_instances_are_deterministic_per_seed_and_differ_between_seeds():
    a = S.make_instance(C.T_COMMUNITIES, 80, 6, 5)
    assert a.adj == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).adj and a.pos == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).pos
    assert S.make_instance(C.T_COMMUNITIES, 80, 6, 6, layout=False).adj != a.adj


def test_the_link_network_is_the_stage_of_the_earlier_pieces():
    full = S.make_instance(C.T_LINK, 0, 0, 0, layout=False)
    assert full.n == 350 and full.m == 2374 and S.link_stand() == "2026-09-30" and "index.html" in full.labels


def test_prepare_calibrates_p_and_draws_independent_snapshot_sets():
    inst = S.make_instance(C.T_RANDOM, 100, 6, 3, layout=False)
    prep = ev.prepare(inst.adj, 10, 3)
    assert prep.comp_t.shape == (C.TRAIN_RUNS, 100) and prep.comp_e.shape == (C.EVAL_RUNS, 100)
    assert not np.array_equal(prep.comp_t[:5], prep.comp_e[:5])                                  # Training und Bewertung sind verschiedene Schnappschüsse
    assert prep.size_e.mean() / 100 == pytest.approx(0.10, abs=0.015)                            # die Ausbreitungsstärke trifft den Zielwert
    again = ev.prepare(inst.adj, 10, 3)
    assert again.p == prep.p and np.array_equal(again.comp_t, prep.comp_t)
    assert ev.prepare(inst.adj, 20, 3).p > prep.p                                                  # stärkere Ausbreitung: größeres p


def test_method_orders_cover_every_method_with_nested_prefixes():
    inst = S.make_instance(C.T_POWER, 80, 6, 4, layout=False)
    prep = ev.prepare(inst.adj, 10, 4)
    orders = ev.method_orders(prep, 6, 4)
    assert set(orders) == set(C.METHODS)
    assert len(orders[C.M_GREEDY]) == 1 and len(orders[C.M_DD_INDEX]) == 1 and len(orders[C.M_DEGREE]) == C.TIE_DRAWS and len(orders[C.M_RANDOM]) == C.RANDOM_DRAWS
    for lst in orders.values():
        for o in lst:
            assert len(o) == 6 and len(set(o)) == 6 and all(0 <= v < 80 for v in o)
    curves = ev.method_curves(prep, orders)
    for m, arr in curves.items():
        assert len(arr) == 6 and all(arr[i] <= arr[i + 1] + 1e-9 for i in range(5)), m           # der Spread wächst mit jedem Starter
    assert orders[C.M_GREEDY][0][:3] == ev.method_orders(prep, 3, 4, methods=(C.M_GREEDY,))[C.M_GREEDY][0]       # die Wahl für k = 3 ist der Anfang der Wahl für k = 6


def test_method_orders_are_deterministic():
    inst = S.make_instance(C.T_GRID, 100, 4, 1, layout=False)
    prep = ev.prepare(inst.adj, 10, 1)
    assert ev.method_orders(prep, 5, 1) == ev.method_orders(prep, 5, 1)


def test_random_tie_resolution_and_the_index_rule_differ_on_the_grid():
    inst = S.make_instance(C.T_GRID, 100, 4, 1, layout=False)
    prep = ev.prepare(inst.adj, 10, 1)
    orders = ev.method_orders(prep, 8, 1, methods=(C.M_DD, C.M_DD_INDEX))
    assert any(o != orders[C.M_DD_INDEX][0] for o in orders[C.M_DD])
    assert len({tuple(o) for o in orders[C.M_DD]}) > 1                                          # die zufälligen Auflösungen liefern verschiedene Mengen


def test_results_do_not_depend_on_the_hash_seed():
    """Dasselbe Szenario in zwei Prozessen mit verschiedenem PYTHONHASHSEED rechnet dasselbe."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import ifm_evaluation as ev, ifm_constants as C\n"
        "out = []\n"
        "for typ, m in ((C.T_RANDOM, C.M_GREEDY), (C.T_POWER, C.M_DD), (C.T_GRID, C.M_CLOSENESS)):\n"
        "    an = ev.analyse(ev.Settings(typ, 80, 6, 7, 10, 4, m))\n"
        "    out.append((an.inst.m, round(an.prep.p, 9), round(an.spread(m), 6), an.chosen, an.celf_evals))\n"
        "print(repr(out))\n"
    ) % str(ROOT)
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1]
