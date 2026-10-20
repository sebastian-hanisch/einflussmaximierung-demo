"""Orakel-Test einflussmaximierung-demo: die Eigenbauten gegen einen anderen Rechenweg (networkx, exakte Aufzählung, Formel von Hand, Optimum durch Aufzählung).

Orakel je Einheit (ORACLE-PLAYBOOK §2): Zusammenhangskomponenten der Schnappschüsse gegen `networkx.connected_components` auf denselben ausgedünnten Netzen; der Schnappschuss-Schätzer
des Spreads gegen den exakten Spread durch Aufzählung aller 2^m Zustände (Band); Greedy gegen das Optimum durch Aufzählung aller Mengen (Garantie 1 − 1/e); CELF gegen naives Greedy;
Submodularität und Monotonie des Schnappschuss-Spreads; Nähe, Betweenness, PageRank und Kernzahl gegen `networkx`; Degree Discount gegen eine unabhängig geschriebene Neuberechnung."""

import itertools

import numpy as np
import pytest

import ifm_algorithm as A
import ifm_constants as C
from conftest import make_adj

nx = pytest.importorskip("networkx")


def _random_adj(seed, lo=5, hi=20):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(lo, hi))
    p = float(rng.uniform(0.1, 0.5))
    return make_adj(n, [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p])


def _graph(adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(adj)))
    g.add_edges_from((a, b) for a in range(len(adj)) for b in adj[a] if a < b)
    return g


@pytest.mark.parametrize("seed", range(40))
def test_snapshot_components_match_networkx_on_the_same_thinned_networks(seed):
    adj = _random_adj(seed)
    n = len(adj)
    u, v = A.edge_arrays(adj)
    p = 0.4
    comp, size = A.snapshot_components(n, u, v, p, 6, seed)
    rng = np.random.default_rng(seed)                                          # derselbe Zufallsstrom wie im Kern: je Schnappschuss m Zufallszahlen
    for r in range(6):
        live = rng.random(len(u)) < p
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(zip(u[live].tolist(), v[live].tolist()))
        parts = {frozenset(c) for c in nx.connected_components(g)}
        mine = {}
        for i in range(n):
            mine.setdefault(int(comp[r, i]), set()).add(i)
        assert {frozenset(s) for s in mine.values()} == parts
        for c in parts:
            assert all(size[r, i] == len(c) for i in c)


def _small_net(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(7, 10))
    m = int(rng.integers(8, 13))
    pairs = set()
    while len(pairs) < m:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        pairs.add((a, b))
    return n, sorted(pairs)


@pytest.mark.parametrize("seed", range(30))
def test_snapshot_estimate_matches_the_exact_spread(seed):
    n, edges = _small_net(seed)
    adj = make_adj(n, edges)
    table = A.exact_table(n, edges, 0.5)
    comp, size = A.snapshot_components(n, *A.edge_arrays(adj), 0.5, 3000, seed)
    for S in ([0], [1, 3], [2, 4, 6]):
        assert A.spread_of(comp, size, S) == pytest.approx(A.exact_spread(table, S), abs=0.2), (seed, S)


def test_greedy_against_the_optimum_by_enumeration_and_the_guarantee():
    ratios = []
    for seed in range(40):
        n, edges = _small_net(seed)
        table = A.exact_table(n, edges, C.SMALL_P)
        for k in C.SMALL_K:
            greedy = A.exact_greedy(table, n, k)
            best, value = A.exact_optimum(table, n, k)
            ratio = A.exact_spread(table, greedy) / value
            assert C.GUARANTEE - 1e-9 <= ratio <= 1 + 1e-9, (seed, k)
            ratios.append(ratio)
    assert min(ratios) >= 0.9 and sum(r > 1 - 1e-9 for r in ratios) >= 10                                         # in der Praxis weit über der Garantie, oft optimal


@pytest.mark.parametrize("seed", range(40))
def test_celf_equals_greedy_on_snapshots(seed):
    adj = _random_adj(seed, 8, 25)
    n = len(adj)
    comp, size = A.snapshot_components(n, *A.edge_arrays(adj), 0.4, 40, seed)
    k = min(5, n)
    g, naive = A.greedy_select(comp, size, k)
    c, lazy = A.celf_select(comp, size, k)
    assert g == c and lazy <= naive and naive == sum(n - j for j in range(k))
    assert A.spread_of(comp, size, g) == A.spread_of(comp, size, c)


def test_snapshot_spread_is_monotone_and_submodular():
    rng = np.random.default_rng(5)
    for seed in range(15):
        adj = _random_adj(seed, 8, 20)
        n = len(adj)
        comp, size = A.snapshot_components(n, *A.edge_arrays(adj), 0.5, 50, seed)
        for _ in range(20):
            big = rng.choice(n, size=min(n - 1, 5), replace=False).tolist()
            small = big[:2]
            v = int(rng.choice([x for x in range(n) if x not in big]))
            f = lambda S: A.spread_of(comp, size, S)
            assert f(big) >= f(small) - 1e-9                                                       # monoton
            assert f(small + [v]) - f(small) >= f(big + [v]) - f(big) - 1e-9                         # submodular: der Zuwachs fällt mit der Menge


@pytest.mark.parametrize("seed", range(60))
def test_measures_match_networkx(seed):
    adj = _random_adj(seed, 5, 30)
    g = _graph(adj)
    clo, bet = nx.closeness_centrality(g), nx.betweenness_centrality(g)
    core = nx.core_number(g)
    pr = nx.pagerank(g, alpha=0.85, tol=1e-12, max_iter=500)
    n = len(adj)
    assert A.closeness(adj) == pytest.approx([clo[i] for i in range(n)], abs=1e-9)
    assert A.betweenness(adj) == pytest.approx([bet[i] for i in range(n)], abs=1e-9)
    assert A.core_numbers(adj) == [core[i] for i in range(n)]
    assert A.pagerank(adj) == pytest.approx([pr[i] for i in range(n)], abs=1e-7)


def _degree_discount_reference(adj, k, p, tie):
    """Unabhängige Fassung: der Wert jedes Knotens wird aus der Menge der gewählten Knoten neu berechnet, nicht fortgeschrieben."""
    n = len(adj)
    chosen = []
    for _ in range(k):
        best, best_key = None, None
        for v in range(n):
            if v in chosen:
                continue
            t = sum(1 for w in adj[v] if w in chosen)
            d = len(adj[v])
            score = d - 2 * t - (d - t) * t * p
            key = (-round(score, 12), float(tie[v]), v)
            if best_key is None or key < best_key:
                best, best_key = v, key
        chosen.append(best)
    return chosen


@pytest.mark.parametrize("seed", range(40))
def test_degree_discount_matches_the_recomputed_formula(seed):
    adj = _random_adj(seed, 6, 30)
    tie = A.tiebreak_vector(len(adj), np.random.default_rng(seed))
    for p in (0.05, 0.3):
        assert A.degree_discount_order(adj, min(6, len(adj)), p, tie) == _degree_discount_reference(adj, min(6, len(adj)), p, tie)


def test_random_tie_resolution_matches_the_enumeration_of_all_orders():
    values = [2, 2, 1, 1]
    seen = set()
    for seed in range(200):
        seen.add(tuple(A.ranking_by(values, A.tiebreak_vector(4, np.random.default_rng(seed)))[:2]))
    assert seen == {(0, 1), (1, 0)}                                                                  # die beiden gleichen Spitzenknoten tauschen die Plätze; der Rest bleibt
    assert {tuple(A.ranking_by(values, A.tiebreak_vector(4, np.random.default_rng(s)))) for s in range(200)} == {(a, b, c, d) for (a, b) in itertools.permutations([0, 1]) for (c, d) in itertools.permutations([2, 3])}
