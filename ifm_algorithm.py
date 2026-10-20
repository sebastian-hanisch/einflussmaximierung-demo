"""Kern: Independent Cascade als Erreichbarkeit in zufällig ausgedünnten Netzen („Schnappschüsse“), Greedy und CELF, Auswahlregeln (Grad, Nähe, Betweenness, PageRank, Kernzahl, Degree Discount,
Single Discount), Abstand und Überlappung der Starter, exakter Spread durch Aufzählung für kleine Netze.

Modell: auf einem ungerichteten Netz gibt jeder Informierte die Nachricht an jeden Nachbarn mit gleicher Wahrscheinlichkeit p weiter, jede Kante bekommt genau eine Chance (Independent Cascade,
Kempe, Kleinberg, Tardos 2003). Das ist gleichbedeutend mit Erreichbarkeit im Netz, in dem jede Kante unabhängig mit Wahrscheinlichkeit p bleibt (Schnappschuss): der erwartete Spread einer
Menge S ist die mittlere Größe der Vereinigung der Zusammenhangskomponenten ihrer Knoten. Greedy rechnet auf festen Trainings-Schnappschüssen (NewGreedyIC, Chen, Wang, Yang 2009), bewertet
wird auf unabhängigen.

Netze sind Nachbarlisten `adj` (Tupel von sortierten Tupeln, Knoten 0 bis n - 1, ungerichtet). Zufall nur über `numpy.random.default_rng(Seed)`."""

import heapq
import itertools
import math

import numpy as np

import ifm_constants as C

ROUND = 12   # Gewinne und Werte werden vor Vergleichen auf 12 Stellen gerundet: Gleitkomma-Summen mathematisch gleicher Knoten unterscheiden sich um ~1e-16


# --- Netze ----------------------------------------------------------------------------------------------------------------------------------------


def edge_arrays(adj):
    """Kanten (a < b) als zwei Arrays."""
    pairs = [(a, b) for a in range(len(adj)) for b in adj[a] if a < b]
    if not pairs:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    u, v = zip(*pairs)
    return np.array(u, dtype=int), np.array(v, dtype=int)


def degrees(adj):
    return [len(a) for a in adj]


def bfs_distances(adj, s):
    dist = {s: 0}
    queue = [s]
    for x in queue:
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return dist


# --- Schnappschüsse und Spread ----------------------------------------------------------------------------------------------------------------------


def snapshot_components(n, u, v, p, runs, seed):
    """`runs` Schnappschüsse: jede Kante bleibt mit Wahrscheinlichkeit p. Rückgabe (comp, size), beide (runs, n): Nummer der Zusammenhangskomponente und Größe der Komponente je Knoten."""
    rng = np.random.default_rng(int(seed))
    m = len(u)
    ul, vl = u.tolist(), v.tolist()
    comp = np.empty((runs, n), dtype=np.int32)
    size = np.empty((runs, n), dtype=np.int32)
    for r in range(runs):
        parent = list(range(n))
        for e in np.flatnonzero(rng.random(m) < p).tolist():
            a, b = ul[e], vl[e]
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[b] = a
        roots = []
        for i in range(n):
            a = i
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            roots.append(a)
        _, lab = np.unique(roots, return_inverse=True)
        comp[r] = lab
        size[r] = np.bincount(lab)[lab]
    return comp, size


def cover_totals(comp, size, order):
    """Je Schnappschuss die Zahl der Knoten, die die Starter `order` erreichen (Vereinigung ihrer Komponenten)."""
    runs, n = comp.shape
    covered = np.zeros((runs, n), dtype=bool)
    rows = np.arange(runs)
    total = np.zeros(runs)
    for node in order:
        lab = comp[:, node]
        total += size[:, node] * ~covered[rows, lab]
        covered[rows, lab] = True
    return total


def prefix_spreads(comp, size, order):
    """Erwarteter Spread der ersten 1, 2, ... Starter von `order` in einem Durchlauf (Liste der Länge len(order))."""
    runs, n = comp.shape
    covered = np.zeros((runs, n), dtype=bool)
    rows = np.arange(runs)
    total = np.zeros(runs)
    out = []
    for node in order:
        lab = comp[:, node]
        total += size[:, node] * ~covered[rows, lab]
        covered[rows, lab] = True
        out.append(float(total.mean()))
    return out


def spread_of(comp, size, order):
    """Erwarteter Spread der Menge `order` (0 für die leere Menge)."""
    return float(cover_totals(comp, size, order).mean()) if len(order) else 0.0


def spread_se(comp, size, order):
    """Standardfehler der Spread-Schätzung (Streuung über die Schnappschüsse geteilt durch √Läufe); 0 bei nur einem Lauf."""
    if comp.shape[0] < 2 or not len(order):
        return 0.0
    return float(cover_totals(comp, size, order).std(ddof=1) / math.sqrt(comp.shape[0]))


def reach_probability(comp, order):
    """Wahrscheinlichkeit jedes Knotens, von den Startern `order` erreicht zu werden (Anteil der Schnappschüsse)."""
    runs, n = comp.shape
    reached = np.zeros((runs, n), dtype=bool)
    for s in order:
        reached |= comp == comp[:, [s]]
    return reached.mean(0)


def single_spreads(size):
    """Erwarteter Spread jedes Knotens allein (mittlere Komponentengröße)."""
    return size.mean(0)


def redundancy(comp, size, order):
    """Spread der Menge geteilt durch die Summe der Einzelspreads: 1 = die Reichweiten überlappen nicht, kleiner = Überlappung."""
    if not len(order):
        return float("nan")
    single = single_spreads(size)
    total = float(sum(single[s] for s in order))
    return spread_of(comp, size, order) / total if total > 0 else float("nan")


def mean_pairwise_distance(adj, order):
    """Mittlerer kürzester Abstand über alle Paare der Starter, die im Netz verbunden sind (nicht erreichbare Paare zählen nicht mit); 0 bei weniger als zwei Startern, `nan` ohne verbundenes Paar."""
    order = list(order)
    if len(order) < 2:
        return 0.0
    dists = [bfs_distances(adj, s) for s in order]
    vals = [dists[i][order[j]] for i in range(len(order)) for j in range(i + 1, len(order)) if order[j] in dists[i]]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def calibrate_p(n, u, v, target, seed=0):
    """p so, dass ein zufälliger Einzelstarter im Mittel `target` · n Knoten erreicht (Bisektion mit CALIB_RUNS Schnappschüssen je Schritt, 18 Schritte)."""
    lo, hi = 0.0, 1.0
    for _ in range(C.CALIB_ITERS):
        mid = (lo + hi) / 2.0
        _, size = snapshot_components(n, u, v, mid, C.CALIB_RUNS, seed)
        if size.mean() / n < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- Greedy und CELF ------------------------------------------------------------------------------------------------------------------------------------


def greedy_select(comp, size, k):
    """Greedy auf festen Schnappschüssen: in jeder Runde der Knoten mit dem größten Zuwachs (Summe über die Schnappschüsse, ganzzahlig), Gleichstand: kleinere Nummer.
    Rückgabe (Starter in Wahlreihenfolge, Zahl der Gewinn-Auswertungen); naiv sind es n - j Auswertungen in Runde j."""
    runs, n = comp.shape
    covered = np.zeros((runs, n), dtype=bool)
    rows = np.arange(runs)
    chosen = []
    evals = 0
    for _ in range(min(k, n)):
        cov_node = np.take_along_axis(covered, comp, axis=1)
        gains = (size.astype(np.int64) * ~cov_node).sum(0)
        gains[chosen] = -1
        evals += n - len(chosen)
        v = int(np.argmax(gains))
        chosen.append(v)
        covered[rows, comp[:, v]] = True
    return chosen, evals


def celf_select(comp, size, k):
    """Lazy Greedy (CELF, Leskovec u. a. 2007): dieselbe Auswahl wie `greedy_select` (der Spread auf festen Schnappschüssen ist exakt submodular, Gewinne fallen nur),
    aber nur so viele Gewinne werden neu gerechnet, bis der oberste Eintrag aktuell ist. Rückgabe (Starter, Zahl der Gewinn-Auswertungen)."""
    runs, n = comp.shape
    covered = np.zeros((runs, n), dtype=bool)
    rows = np.arange(runs)
    evals = 0

    def gain(v):
        nonlocal evals
        evals += 1
        return int((size[:, v].astype(np.int64) * ~covered[rows, comp[:, v]]).sum())

    heap = [(-gain(v), v, 0) for v in range(n)]
    heapq.heapify(heap)
    chosen = []
    while len(chosen) < min(k, n):
        g, v, stamp = heapq.heappop(heap)
        if stamp == len(chosen):
            chosen.append(v)
            covered[rows, comp[:, v]] = True
        else:
            heapq.heappush(heap, (-gain(v), v, len(chosen)))
    return chosen, evals


# --- Maße für die Auswahlregeln -----------------------------------------------------------------------------------------------------------------------------


def closeness(adj):
    """Nähe nach Wasserman und Faust (wie `networkx.closeness_centrality`): (r - 1) / Summe der Abstände, mit (r - 1) / (n - 1) skaliert (r = erreichbare Knoten einschließlich des Knotens)."""
    n = len(adj)
    out = []
    for s in range(n):
        dist = bfs_distances(adj, s)
        total = sum(dist.values())
        r = len(dist)
        out.append(0.0 if total == 0 or n <= 1 else (r - 1) / total * (r - 1) / (n - 1))
    return out


def betweenness(adj):
    """Betweenness nach Brandes (2001), normiert wie `networkx.betweenness_centrality`."""
    n = len(adj)
    cb = [0.0] * n
    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = [0.0] * n
        sigma[s] = 1.0
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        for x in queue:
            stack.append(x)
            for w in adj[x]:
                if dist[w] < 0:
                    dist[w] = dist[x] + 1
                    queue.append(w)
                if dist[w] == dist[x] + 1:
                    sigma[w] += sigma[x]
                    pred[w].append(x)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for x in pred[w]:
                delta[x] += sigma[x] / sigma[w] * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    return [x * scale for x in cb]


def pagerank(adj, damping=0.85, tol=1e-10, max_iter=200):
    """PageRank (Potenziteration, ungerichtet, Knoten ohne Kontakt verteilen gleichmäßig)."""
    n = len(adj)
    if n == 0:
        return []
    deg = np.array([len(a) for a in adj], dtype=float)
    rank = np.full(n, 1.0 / n)
    src = np.array([a for a in range(n) for _ in adj[a]], dtype=int)
    dst = np.array([b for a in range(n) for b in adj[a]], dtype=int)
    for _ in range(max_iter):
        contrib = np.zeros(n)
        share = np.divide(rank, deg, out=np.zeros(n), where=deg > 0)
        np.add.at(contrib, dst, share[src])
        dangling = rank[deg == 0].sum()
        new = (1.0 - damping) / n + damping * (contrib + dangling / n)
        if np.abs(new - rank).sum() < tol:
            rank = new
            break
        rank = new
    return [float(x) for x in rank]


def core_numbers(adj):
    """Kernzahl je Knoten durch Schälen: für k = 0, 1, ... werden Knoten mit höchstens k verbliebenen Nachbarn entfernt (und ihre Nachbarn abgewertet), bis nichts mehr geht."""
    n = len(adj)
    deg = [len(a) for a in adj]
    removed = [False] * n
    core = [0] * n
    remaining = n
    k = 0
    while remaining:
        stack = [v for v in range(n) if not removed[v] and deg[v] <= k]
        if not stack:
            k += 1
            continue
        while stack:
            v = stack.pop()
            if removed[v]:
                continue
            removed[v] = True
            core[v] = k
            remaining -= 1
            for w in adj[v]:
                if not removed[w]:
                    deg[w] -= 1
                    if deg[w] <= k:
                        stack.append(w)
    return core


# --- Auswahlregeln ----------------------------------------------------------------------------------------------------------------------------------------


def tiebreak_vector(n, rng=None):
    """Rangfolge der Knoten bei Gleichstand: zufällig (`rng`) oder nach Knotennummer (ohne `rng`)."""
    return rng.random(n) if rng is not None else np.arange(n, dtype=float)


def ranking_by(values, tie):
    """Alle Knoten nach Wert absteigend (auf 12 Stellen gerundet), Gleichstand nach `tie` (kleinerer Wert zuerst)."""
    return sorted(range(len(values)), key=lambda i: (-round(float(values[i]), ROUND), float(tie[i]), i))


def degree_discount_order(adj, k, p, tie):
    """DegreeDiscountIC (Chen, Wang, Yang 2009): dd_v = d_v - 2 t_v - (d_v - t_v) t_v p, t_v = Zahl der schon gewählten Nachbarn; nach jeder Wahl werden die Nachbarn neu bewertet."""
    n = len(adj)
    d = [len(a) for a in adj]
    t = [0] * n
    dd = [float(x) for x in d]
    chosen = []
    taken = [False] * n
    for _ in range(min(k, n)):
        v = min((i for i in range(n) if not taken[i]), key=lambda i: (-round(dd[i], ROUND), float(tie[i]), i))
        chosen.append(v)
        taken[v] = True
        for w in adj[v]:
            if not taken[w]:
                t[w] += 1
                dd[w] = d[w] - 2 * t[w] - (d[w] - t[w]) * t[w] * p
    return chosen


def single_discount_order(adj, k, tie):
    """SingleDiscount: jeder Nachbar eines gewählten Starters verliert genau einen Kontakt."""
    n = len(adj)
    dd = [float(len(a)) for a in adj]
    chosen = []
    taken = [False] * n
    for _ in range(min(k, n)):
        v = min((i for i in range(n) if not taken[i]), key=lambda i: (-round(dd[i], ROUND), float(tie[i]), i))
        chosen.append(v)
        taken[v] = True
        for w in adj[v]:
            if not taken[w]:
                dd[w] -= 1
    return chosen


# --- Exakter Spread für kleine Netze (Aufzählung aller 2^m Zustände) ------------------------------------------------------------------------------------


def exact_table(n, edges, p):
    """Alle 2^m Zustände der Kanten (m <= 16) mit Wahrscheinlichkeit und Komponenten: Rückgabe (probs, comp, size) mit Formen (Z,), (Z, n), (Z, n)."""
    m = len(edges)
    if m > 16:
        raise ValueError("zu viele Kanten für die Aufzählung")
    states = 1 << m
    probs = np.empty(states)
    comp = np.empty((states, n), dtype=np.int32)
    size = np.empty((states, n), dtype=np.int32)
    for s in range(states):
        live = [edges[i] for i in range(m) if s >> i & 1]
        parent = list(range(n))
        for a, b in live:
            while parent[a] != a:
                a = parent[a]
            while parent[b] != b:
                b = parent[b]
            if a != b:
                parent[b] = a
        roots = []
        for i in range(n):
            a = i
            while parent[a] != a:
                a = parent[a]
            roots.append(a)
        _, lab = np.unique(roots, return_inverse=True)
        comp[s] = lab
        size[s] = np.bincount(lab)[lab]
        probs[s] = p ** len(live) * (1.0 - p) ** (m - len(live))
    return probs, comp, size


def exact_spread(table, order):
    """Exakter erwarteter Spread der Menge `order` aus einer Tabelle von `exact_table`."""
    probs, comp, size = table
    if not len(order):
        return 0.0
    return float((probs * cover_totals(comp, size, order)).sum())


def exact_greedy(table, n, k):
    """Greedy mit dem exakten Spread (Gleichstand: kleinere Nummer); Rückgabe die Starter in Wahlreihenfolge."""
    chosen = []
    for _ in range(min(k, n)):
        gains = [(-1.0 if v in chosen else exact_spread(table, chosen + [v])) for v in range(n)]
        chosen.append(int(np.argmax(np.round(gains, ROUND))))
    return chosen


def exact_optimum(table, n, k):
    """Beste Menge aus k Knoten durch Aufzählung aller Mengen; Rückgabe (Menge, Spread); bei Gleichstand die zuerst gefundene (lexikographisch kleinste)."""
    best, best_set = -1.0, ()
    for combo in itertools.combinations(range(n), min(k, n)):
        value = exact_spread(table, list(combo))
        if round(value, ROUND) > round(best, ROUND):
            best, best_set = value, combo
    return list(best_set), best
