"""Gemeinsame Fixtures und Hilfen (DEMO-PLAYBOOK §1b): Mini-Instanzen, an denen jede Einheit von Hand geprüft wird."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_adj(n, pairs):
    """Nachbarlisten (Tupel von sortierten Tupeln) aus einer Kantenliste."""
    sets = [set() for _ in range(n)]
    for a, b in pairs:
        if a != b:
            sets[a].add(b)
            sets[b].add(a)
    return tuple(tuple(sorted(s)) for s in sets)


@pytest.fixture
def path3():
    """Pfad 0 - 1 - 2, p = 0,5. Handrechnung: σ({1}) = 1 + 0,5 + 0,5 = 2; σ({0}) = 1 + 0,5 (Knoten 1) + 0,25 (Knoten 2) = 1,75."""
    return make_adj(3, [(0, 1), (1, 2)])


@pytest.fixture
def star3():
    """Stern: Mitte 0, Blätter 1, 2, 3. Bei p = 1 erreicht jeder Starter alle vier Knoten (Max-Coverage-Spezialfall)."""
    return make_adj(4, [(0, 1), (0, 2), (0, 3)])


@pytest.fixture
def two_triangles():
    """Zwei getrennte Dreiecke {0,1,2} und {3,4,5}; bei p = 1 reicht jeder Starter genau bis in sein Dreieck."""
    return make_adj(6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])


@pytest.fixture
def discount_net():
    """Hub 0 (Grad 4: 1, 2, 8, 9), Knoten 1 (Grad 3: 0, 3, 4) und ein zweiter Stern 5 (Grad 2: 6, 7).
    Grad wählt 0, 1, 5. Degree Discount mit p = 0,1 wertet 1 nach der Wahl von 0 auf 3 - 2 - (3 - 1) · 1 · 0,1 = 0,8 ab und wählt 5 (Grad 2) davor."""
    return make_adj(10, [(0, 1), (0, 2), (0, 8), (0, 9), (1, 3), (1, 4), (5, 6), (5, 7)])
