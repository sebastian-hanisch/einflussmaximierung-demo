"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): k = 1, k = n, kleinste und größte Regler, Gleichstände, kleine Netze."""

import numpy as np
import pytest

import ifm_algorithm as A
import ifm_constants as C
import ifm_evaluation as ev
from conftest import make_adj


def test_k_equal_to_one_and_to_n_on_snapshots(path3):
    comp, size = A.snapshot_components(3, *A.edge_arrays(path3), 0.5, 200, 1)
    assert A.greedy_select(comp, size, 1)[0] == [1] and A.celf_select(comp, size, 1)[0] == [1]                  # die Mitte reicht am weitesten
    full, _ = A.greedy_select(comp, size, 3)
    assert sorted(full) == [0, 1, 2] and A.spread_of(comp, size, full) == pytest.approx(3.0)                   # alle Knoten als Starter: jeder Knoten ist erreicht
    assert A.celf_select(comp, size, 3)[0] == full


def test_network_without_edges_has_spread_k():
    adj = make_adj(5, [])
    u, v = A.edge_arrays(adj)
    comp, size = A.snapshot_components(5, u, v, 0.5, 20, 1)
    assert A.greedy_select(comp, size, 2)[0] == [0, 1] and A.spread_of(comp, size, [0, 1]) == 2.0 and A.redundancy(comp, size, [0, 1]) == pytest.approx(1.0)
    assert 0.0 <= A.calibrate_p(5, u, v, 0.3, 1) <= 1.0


def test_spread_is_k_at_p_zero_and_component_size_at_p_one(two_triangles):
    u, v = A.edge_arrays(two_triangles)
    comp, size = A.snapshot_components(6, u, v, 0.0, 10, 1)
    assert A.spread_of(comp, size, [0, 3]) == 2.0 and A.spread_of(comp, size, [0]) == 1.0
    comp, size = A.snapshot_components(6, u, v, 1.0, 10, 1)
    assert A.spread_of(comp, size, [0, 3]) == 6.0


def test_analysis_of_every_type_at_the_slider_corners_does_not_fail():
    for typ in C.GENERATED_TYPES:
        for n, c, strength, k in ((C.N_MIN, C.C_MIN, C.STRENGTH_MAX, C.K_MAX), (C.N_MAX, C.C_MAX, C.STRENGTH_MIN, C.K_MIN)):
            an = ev.analyse(ev.Settings(typ, n, c, 1, strength, k, C.M_DD_INDEX))
            assert an.k == min(k, an.inst.n) and len(an.chosen) == an.k and len(set(an.chosen)) == an.k
            assert an.reach.min() >= 0.0 and an.reach.max() <= 1.0 and all(an.reach[s] == 1.0 for s in an.chosen)
            assert ev.verdict_main(an)[0] in ("success", "info", "warning")
    for method in C.METHODS:
        an = ev.analyse(ev.Settings(C.T_COMMUNITIES, 80, 6, 2, 10, 4, method))
        assert an.method == method and an.spread(method) >= an.k - 1e-9


def test_k_larger_than_the_network_is_limited():
    an = ev.analyse(ev.Settings(C.T_COMMUNITIES, C.N_MIN, 6, 1, 10, C.K_MAX))
    assert an.k == C.K_MAX <= C.N_MIN                                                                      # der Slider erlaubt nie mehr Starter als Knoten
    adj = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    prep = ev.prepare(adj, 30, 1, train=20, evaluate=20)
    assert len(ev.method_orders(prep, 9, 1)[C.M_GREEDY][0]) == 4


def test_calibration_is_robust_at_the_strength_bounds():
    inst = ev.make_instance(C.T_RANDOM, 100, 6, 2, layout=False)
    low, high = ev.prepare(inst.adj, C.STRENGTH_MIN, 2), ev.prepare(inst.adj, C.STRENGTH_MAX, 2)
    assert 0 < low.p < high.p < 1
    assert low.size_e.mean() / 100 == pytest.approx(0.03, abs=0.015) and high.size_e.mean() / 100 == pytest.approx(0.30, abs=0.04)


def test_experiments_survive_tiny_networks():
    n = C.N_MIN
    assert len(ev.curve_rows(n, C.C_MIN, C.STRENGTH_MIN, seeds=(1,))) == len(C.GENERATED_TYPES)
    assert len(ev.distance_rows(n, C.C_MIN, C.STRENGTH_MAX, 1, seeds=(1,))) == len(C.GENERATED_TYPES)
    assert len(ev.celf_rows(n, C.C_MIN, C.STRENGTH_MIN, seeds=(1,))) == len(C.GENERATED_TYPES)
    assert len(ev.noise_rows(C.T_GRID, n, C.C_MIN, C.STRENGTH_MIN, 3, seeds=(1,))) == len(C.NOISE_TRAIN)
    assert set(ev.guarantee_rows(nets=3)["ratios"]) == set(C.SMALL_K)


def test_distance_rows_for_a_single_starter_are_zero():
    rows = ev.distance_rows(C.N_MIN, 4, 10, 1, seeds=(1,), types=(C.T_RANDOM,))
    assert rows[0][C.M_GREEDY]["dist"] == 0.0 and rows[0][C.M_GREEDY]["red"] == pytest.approx(1.0)        # ein Starter: keine Paare, keine Überlappung


def test_small_networks_are_valid_and_deterministic():
    for i in range(30):
        n, edges = ev.small_network(100000 + i)
        assert C.SMALL_N[0] <= n <= C.SMALL_N[1] and C.SMALL_M[0] <= len(edges) <= C.SMALL_M[1] and len(set(edges)) == len(edges)
        assert all(0 <= a < b < n for a, b in edges)
        assert (n, edges) == ev.small_network(100000 + i)


def test_the_grid_has_ties_so_the_tie_rule_matters():
    inst = ev.make_instance(C.T_GRID, 100, 4, 1, layout=False)
    deg = A.degrees(inst.adj)
    assert sum(1 for d in deg if d == 4) > 10                                                               # viele Knoten mit demselben Grad
    by_index = A.ranking_by([float(d) for d in deg], A.tiebreak_vector(100))[:5]
    by_chance = {tuple(A.ranking_by([float(d) for d in deg], A.tiebreak_vector(100, np.random.default_rng(s)))[:5]) for s in range(10)}
    assert by_index == sorted(i for i, d in enumerate(deg) if d == 4)[:5] and len(by_chance) > 1
