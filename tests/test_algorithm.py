"""Einheiten des Kerns an Mini-Instanzen mit von Hand gerechneten Werten (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import ifm_algorithm as A
import ifm_constants as C
from conftest import make_adj


def _edges(adj):
    u, v = A.edge_arrays(adj)
    return list(zip(u.tolist(), v.tolist()))


def test_edge_arrays_and_degrees(star3):
    u, v = A.edge_arrays(star3)
    assert u.tolist() == [0, 0, 0] and v.tolist() == [1, 2, 3] and A.degrees(star3) == [3, 1, 1, 1]
    u, v = A.edge_arrays(make_adj(3, []))
    assert len(u) == 0 and len(v) == 0


def test_exact_spread_hand_values_on_the_path(path3):
    table = A.exact_table(3, _edges(path3), 0.5)
    assert table[0].sum() == pytest.approx(1.0) and len(table[0]) == 4
    assert A.exact_spread(table, [1]) == pytest.approx(2.0) and A.exact_spread(table, [0]) == pytest.approx(1.75) and A.exact_spread(table, [2]) == pytest.approx(1.75)
    assert A.exact_spread(table, []) == 0.0


def test_exact_spread_of_two_starters_on_the_path_by_hand(path3):
    # Zustände der Kanten (01, 12): beide live (p²= 1/4): alle 3 erreicht; nur 01 (1/4): {0,1} und {2} → Starter 0 und 2 erreichen 3; nur 12 (1/4): ebenso 3; keine (1/4): 2 Starter = 2 Knoten
    table = A.exact_table(3, _edges(path3), 0.5)
    assert A.exact_spread(table, [0, 2]) == pytest.approx(0.25 * 3 + 0.25 * 3 + 0.25 * 3 + 0.25 * 2)


def test_star_at_p_one_and_two_triangles(star3, two_triangles):
    table = A.exact_table(4, _edges(star3), 1.0)
    assert all(A.exact_spread(table, [v]) == pytest.approx(4.0) for v in range(4))
    comp, size = A.snapshot_components(6, *A.edge_arrays(two_triangles), 1.0, 5, 1)
    assert (size == 3).all() and A.spread_of(comp, size, [0]) == 3.0 and A.spread_of(comp, size, [0, 3]) == 6.0 and A.spread_of(comp, size, [0, 1]) == 3.0
    order, evals = A.greedy_select(comp, size, 2)
    assert order == [0, 3]                                                                  # erst Knoten 0 (Gleichstand: kleinste Nummer), dann der erste Knoten des anderen Dreiecks
    assert A.redundancy(comp, size, [0, 3]) == pytest.approx(1.0) and A.redundancy(comp, size, [0, 1]) == pytest.approx(0.5)       # 3 / (3 + 3) bei überlappender Reichweite


def test_snapshots_at_p_zero_and_one():
    adj = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    u, v = A.edge_arrays(adj)
    comp, size = A.snapshot_components(4, u, v, 0.0, 6, 1)
    assert (size == 1).all() and all(len(set(row.tolist())) == 4 for row in comp)
    comp, size = A.snapshot_components(4, u, v, 1.0, 6, 1)
    assert (size == 4).all() and all(len(set(row.tolist())) == 1 for row in comp)
    comp, size = A.snapshot_components(3, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 0.5, 3, 1)       # ohne Kanten: jeder Knoten allein
    assert (size == 1).all()


def test_snapshots_are_deterministic_and_independent_of_the_seed_stream(path3):
    u, v = A.edge_arrays(path3)
    a = A.snapshot_components(3, u, v, 0.5, 50, 7)
    b = A.snapshot_components(3, u, v, 0.5, 50, 7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(A.snapshot_components(3, u, v, 0.5, 50, 8)[0], a[0])


def test_prefix_spreads_cover_totals_and_standard_error(two_triangles):
    comp, size = A.snapshot_components(6, *A.edge_arrays(two_triangles), 1.0, 4, 1)
    assert A.prefix_spreads(comp, size, [0, 1, 3]) == [3.0, 3.0, 6.0]
    assert A.cover_totals(comp, size, [0, 3]).tolist() == [6.0] * 4 and A.spread_of(comp, size, []) == 0.0
    assert A.spread_se(comp, size, [0, 3]) == 0.0 and A.spread_se(comp[:1], size[:1], [0]) == 0.0
    u, v = A.edge_arrays(make_adj(3, [(0, 1), (1, 2)]))
    c2, s2 = A.snapshot_components(3, u, v, 0.5, 200, 3)
    assert A.spread_se(c2, s2, [1]) > 0.0                                                      # Standardfehler > 0 bei 0 < p < 1


def test_reach_probability_and_single_spreads(path3):
    u, v = A.edge_arrays(path3)
    comp, size = A.snapshot_components(3, u, v, 1.0, 5, 1)
    assert A.reach_probability(comp, [1]).tolist() == [1.0, 1.0, 1.0]
    comp, size = A.snapshot_components(3, u, v, 0.0, 5, 1)
    assert A.reach_probability(comp, [1]).tolist() == [0.0, 1.0, 0.0] and A.single_spreads(size).tolist() == [1.0, 1.0, 1.0]
    comp, size = A.snapshot_components(3, u, v, 0.5, 4000, 2)
    assert A.reach_probability(comp, [1]).tolist() == pytest.approx([0.5, 1.0, 0.5], abs=0.04)
    assert A.single_spreads(size).tolist() == pytest.approx([1.75, 2.0, 1.75], abs=0.08)


def test_greedy_and_celf_choose_the_same_and_count_evaluations(two_triangles):
    comp, size = A.snapshot_components(6, *A.edge_arrays(two_triangles), 0.7, 60, 3)
    g, naive = A.greedy_select(comp, size, 3)
    c, lazy = A.celf_select(comp, size, 3)
    assert g == c and naive == 6 + 5 + 4 and 6 <= lazy <= naive                                  # naiv: n + (n - 1) + (n - 2) Auswertungen
    assert A.greedy_select(comp, size, 0) == ([], 0) and A.celf_select(comp, size, 0)[0] == []
    assert A.greedy_select(comp, size, 99)[0] == A.celf_select(comp, size, 99)[0] and len(A.greedy_select(comp, size, 99)[0]) == 6      # k größer als n wird begrenzt


def test_greedy_breaks_gain_ties_by_the_smaller_node_number(star3):
    comp, size = A.snapshot_components(4, *A.edge_arrays(star3), 1.0, 3, 1)
    assert A.greedy_select(comp, size, 1)[0] == [0] and A.celf_select(comp, size, 1)[0] == [0]      # alle Gewinne gleich: Knoten 0


def test_calibrate_p_hits_the_target_and_is_monotone():
    adj = make_adj(40, [(i, (i + d) % 40) for i in range(40) for d in (1, 2, 3)])
    u, v = A.edge_arrays(adj)
    ps = [A.calibrate_p(40, u, v, t, 1) for t in (0.1, 0.3, 0.6)]
    assert ps[0] < ps[1] < ps[2]
    _, size = A.snapshot_components(40, u, v, ps[1], 400, 9)
    assert size.mean() / 40 == pytest.approx(0.3, abs=0.06)
    assert A.calibrate_p(40, u, v, 0.9, 1) > ps[2] and A.calibrate_p(40, u, v, 0.0, 1) < 0.05


def test_degree_discount_and_single_discount_by_hand(discount_net):
    index_tie = A.tiebreak_vector(10)
    assert A.ranking_by(A.degrees(discount_net), index_tie)[:3] == [0, 1, 5]                                    # Grad: 4, 3, 2
    assert A.degree_discount_order(discount_net, 3, 0.1, index_tie)[:2] == [0, 5]                              # 1 wird auf 0,8 abgewertet, 5 hat 2
    assert A.degree_discount_order(discount_net, 2, 0.1, index_tie) == [0, 5]
    assert A.single_discount_order(discount_net, 2, index_tie) == [0, 1]                                        # 1 verliert einen Kontakt: 2, Gleichstand mit 5, die kleinere Nummer gewinnt
    prefer_5 = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
    assert A.single_discount_order(discount_net, 2, prefer_5) == [0, 5]                                          # mit anderer Gleichstandsregel anders


def test_ranking_and_tie_breaks():
    assert A.ranking_by([3, 1, 3], A.tiebreak_vector(3)) == [0, 2, 1]
    assert A.ranking_by([3, 1, 3], np.array([0.9, 0.5, 0.1])) == [2, 0, 1]
    assert A.ranking_by([0.3, 0.3 + 3e-16, 0.1], A.tiebreak_vector(3)) == [0, 1, 2]                           # Rauschen von 1e-16 ist ein Gleichstand
    rng = np.random.default_rng(1)
    t1, t2 = A.tiebreak_vector(5, rng), A.tiebreak_vector(5, np.random.default_rng(1))
    assert np.array_equal(t1, t2) and A.tiebreak_vector(3).tolist() == [0.0, 1.0, 2.0]


def test_core_numbers_hand_values():
    triangle_tail = make_adj(4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    assert A.core_numbers(triangle_tail) == [2, 2, 2, 1]
    k4 = make_adj(4, [(a, b) for a in range(4) for b in range(a + 1, 4)])
    assert A.core_numbers(k4) == [3, 3, 3, 3] and A.core_numbers(make_adj(3, [(0, 1), (1, 2)])) == [1, 1, 1] and A.core_numbers(make_adj(3, [])) == [0, 0, 0]


def test_closeness_betweenness_pagerank_hand_values(star3):
    path = make_adj(3, [(0, 1), (1, 2)])
    assert A.closeness(path) == pytest.approx([2 / 3, 1.0, 2 / 3]) and A.betweenness(star3) == pytest.approx([1.0, 0, 0, 0]) and A.betweenness(make_adj(2, [(0, 1)])) == [0.0, 0.0]
    cycle = make_adj(5, [(i, (i + 1) % 5) for i in range(5)])
    assert A.pagerank(cycle) == pytest.approx([0.2] * 5)
    pr = A.pagerank(star3)
    assert sum(pr) == pytest.approx(1.0) and pr[0] > pr[1] and pr[1] == pytest.approx(pr[2])
    assert A.pagerank(make_adj(3, [])) == pytest.approx([1 / 3] * 3) and A.pagerank(()) == []


def test_mean_pairwise_distance_hand_values():
    path = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    assert A.mean_pairwise_distance(path, [0, 3]) == 3.0 and A.mean_pairwise_distance(path, [0, 1, 3]) == pytest.approx((1 + 3 + 2) / 3)
    assert A.mean_pairwise_distance(path, [2]) == 0.0 and A.mean_pairwise_distance(path, []) == 0.0
    apart = make_adj(4, [(0, 1)])
    assert math.isnan(A.mean_pairwise_distance(apart, [0, 3]))                                                 # kein verbundenes Paar: nicht definiert
    assert A.mean_pairwise_distance(apart, [0, 1, 3]) == 1.0                                                   # das nicht erreichbare Paar zählt nicht mit


def test_redundancy_of_an_empty_set_is_undefined(two_triangles):
    comp, size = A.snapshot_components(6, *A.edge_arrays(two_triangles), 1.0, 3, 1)
    assert math.isnan(A.redundancy(comp, size, []))


def test_exact_optimum_and_greedy_on_the_path(path3):
    table = A.exact_table(3, _edges(path3), 0.5)
    best, value = A.exact_optimum(table, 3, 1)
    assert best == [1] and value == pytest.approx(2.0)
    assert A.exact_greedy(table, 3, 1) == [1]
    best2, value2 = A.exact_optimum(table, 3, 2)
    greedy2 = A.exact_greedy(table, 3, 2)
    assert A.exact_spread(table, greedy2) >= C.GUARANTEE * value2 - 1e-12 and A.exact_spread(table, greedy2) <= value2 + 1e-12
    with pytest.raises(ValueError):
        A.exact_table(20, [(i, i + 1) for i in range(17)], 0.5)


def test_exact_greedy_and_optimum_consider_the_last_node_and_break_ties_by_the_first(star3):
    table = A.exact_table(4, [(0, 3), (1, 3), (2, 3)], 0.5)                                   # Mitte ist der letzte Knoten
    assert A.exact_greedy(table, 4, 1) == [3] and A.exact_optimum(table, 4, 1)[0] == [3]
    triangle = A.exact_table(3, [(0, 1), (0, 2), (1, 2)], 0.5)
    assert A.exact_optimum(triangle, 3, 1)[0] == [0] and A.exact_greedy(triangle, 3, 1) == [0]  # alles gleich: die kleinste Nummer


def test_pagerank_of_isolated_nodes_has_no_numpy_warning_and_redundancy_of_a_single_isolated_starter():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert A.pagerank(make_adj(3, [])) == pytest.approx([1 / 3] * 3)
        assert A.pagerank(make_adj(4, [(0, 1)])) == pytest.approx(A.pagerank(make_adj(4, [(0, 1)])))
    comp, size = A.snapshot_components(3, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 0.5, 5, 1)
    assert A.redundancy(comp, size, [0]) == pytest.approx(1.0)                              # ein einzelner Starter ohne Kontakte: Spread 1 geteilt durch 1


def test_exact_table_accepts_sixteen_edges_and_rejects_seventeen():
    path16 = [(i, i + 1) for i in range(16)]                                                   # 16 Kanten: 65 536 Zustände, die Obergrenze
    probs, comp, size = A.exact_table(17, path16, 0.5)
    assert len(probs) == 65536 and probs.sum() == pytest.approx(1.0) and size[-1].tolist() == [17] * 17                      # letzter Zustand: alle Kanten live
    with pytest.raises(ValueError):
        A.exact_table(18, [(i, i + 1) for i in range(17)], 0.5)


def test_standard_error_with_exactly_two_snapshots():
    comp = np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int32)                                    # Schnappschuss 1: eine Komponente (3), Schnappschuss 2: drei einzelne (1)
    size = np.array([[3, 3, 3], [1, 1, 1]], dtype=np.int32)
    assert A.cover_totals(comp, size, [0]).tolist() == [3.0, 1.0]
    assert A.spread_se(comp, size, [0]) == pytest.approx(1.0)                                  # Standardabweichung √2 geteilt durch √2
