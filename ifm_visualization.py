"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import plotly.graph_objects as go

import ifm_algorithm as A
import ifm_constants as C
from ifm_evaluation import de, ratio_at


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, bottom=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=bottom), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _name(inst, i):
    return inst.labels[i] if inst.labels else str(i)


def build_network(inst, reach, chosen, method, height=430):
    """Netz mit festen Positionen: Knoten nach der Wahrscheinlichkeit gefärbt, von den Startern erreicht zu werden, die Starter groß und rot umrandet (Nummer = Wahlreihenfolge bei Greedy)."""
    n = inst.n
    xs, ys = [], []
    u, v = A.edge_arrays(inst.adj)
    for a, b in zip(u.tolist(), v.tolist()):
        xs += [inst.pos[a][0], inst.pos[b][0], None]
        ys += [inst.pos[a][1], inst.pos[b][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=0.7), hoverinfo="skip", showlegend=False))
    size = 5 if n > 200 else 8
    fig.add_trace(go.Scatter(x=[p[0] for p in inst.pos], y=[p[1] for p in inst.pos], mode="markers",
                             marker=dict(size=size, color=[float(r) for r in reach], colorscale="Viridis", cmin=0, cmax=1, showscale=True, colorbar=dict(title=dict(text="erreicht"), thickness=12, len=0.8),
                                         line=dict(color="rgba(40,40,40,0.45)", width=0.4)),
                             hovertext=[f"Knoten {_name(inst, i)}: erreicht mit {100 * float(reach[i]):.0f} %" for i in range(n)], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[inst.pos[s][0] for s in chosen], y=[inst.pos[s][1] for s in chosen], mode="markers+text", text=[str(i + 1) for i in range(len(chosen))], textposition="top center",
                             marker=dict(size=size + 9, color="rgba(0,0,0,0)", line=dict(color=C.COLOR_SELECT, width=2.5)),
                             hovertext=[f"Starter {i + 1}: Knoten {_name(inst, s)}" for i, s in enumerate(chosen)], hoverinfo="text", name=f"Starter ({method})"))
    fig.add_trace(go.Scatter(x=[-0.04, 1.04], y=[-0.04, 1.04], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, autorange=True)
    fig.update_yaxes(visible=False, autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.02), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_methods(an):
    """Erwarteter Spread aller Verfahren bei k Startern (Balken); Greedy mit Standardfehler als Linie, Zufall als Grundlinie."""
    names = list(C.METHODS)
    ys = [an.spread(m) for m in names]
    fig = go.Figure(go.Bar(x=names, y=ys, marker_color=[C.METHOD_COLORS[m] for m in names], text=[de(y, 1) for y in ys], textposition="outside",
                           error_y=dict(type="data", array=[an.se_greedy if m == C.M_GREEDY else 0 for m in names], visible=True)))
    fig.add_hline(y=an.spread(C.M_GREEDY), line=dict(color=C.COLOR_A, dash="dash"), annotation_text="Greedy", annotation_position="top left")
    fig.update_yaxes(title_text=f"erwartete erreichte Knoten (von {an.inst.n})", rangemode="tozero")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=360, bottom=110)


def _color(m):
    return C.METHOD_COLORS[m]


def build_curve(row):
    """Verhältnis von Spread und Greedy-Spread über die Zahl der Starter für alle Verfahren in einem Netztyp."""
    fig = go.Figure()
    for m in C.METHODS:
        if m == C.M_GREEDY:
            continue
        fig.add_trace(go.Scatter(x=row["ks"], y=row["ratios"][m], mode="lines+markers", name=m, line=dict(color=_color(m), width=3 if m in (C.M_DEGREE, C.M_CLOSENESS, C.M_DD_INDEX) else 1.8,
                                                                                               dash="dot" if m == C.M_DD_INDEX else "solid")))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_A, dash="dash"), annotation_text="Greedy", annotation_position="top left")
    fig.update_xaxes(title_text="Zahl der Starter k", tickmode="array", tickvals=row["ks"])
    fig.update_yaxes(title_text="Spread / Spread von Greedy", range=[0.3, 1.1])
    return _base(fig, height=380, bottom=60)


def build_ratio_bars(rows, k):
    """Verhältnis zu Greedy bei k Startern je Netztyp für Grad, Nähe, Kernzahl und Degree Discount (beide Gleichstandsregeln)."""
    methods = (C.M_DEGREE, C.M_CLOSENESS, C.M_CORE, C.M_DD, C.M_DD_INDEX)
    fig = go.Figure()
    for m in methods:
        ys = [ratio_at(r, m, k) for r in rows]
        fig.add_trace(go.Bar(x=[r["typ"] for r in rows], y=ys, name=m, marker_color=_color(m), text=[de(y, 2) for y in ys], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_A, dash="dash"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Spread / Greedy bei k = {k}", range=[0, 1.2])
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=400, bottom=100)


def build_distance(rows):
    """Mittlerer Abstand der Starter (Balken) und Redundanz (Punkte) für Greedy, Grad und Nähe je Netztyp."""
    fig = go.Figure()
    for m in (C.M_GREEDY, C.M_DEGREE, C.M_CLOSENESS):
        fig.add_trace(go.Bar(x=[r["typ"] for r in rows], y=[r[m]["dist"] for r in rows], name=m, marker_color=_color(m), text=[de(r[m]["dist"], 1) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="mittlerer Abstand der Starter")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=380, bottom=100)


def build_redundancy(rows):
    """Redundanz (Spread geteilt durch die Summe der Einzelspreads; 1 = keine Überlappung) für Greedy, Grad und Nähe je Netztyp."""
    fig = go.Figure()
    for m in (C.M_GREEDY, C.M_DEGREE, C.M_CLOSENESS):
        fig.add_trace(go.Bar(x=[r["typ"] for r in rows], y=[r[m]["red"] for r in rows], name=m, marker_color=_color(m), text=[de(r[m]["red"], 2) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Redundanz", range=[0, 1.1])
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=340, bottom=100)


def build_celf(rows, ks=C.CELF_KS):
    """Gewinn-Auswertungen von CELF und naivem Greedy je Netztyp."""
    fig = go.Figure()
    for k in ks:
        fig.add_trace(go.Bar(x=[r["typ"] for r in rows], y=[r["cells"][k]["naive"] for r in rows], name=f"naiv, k = {k}", marker_color=C.COLOR_B, opacity=0.5 if k == ks[0] else 0.9))
        fig.add_trace(go.Bar(x=[r["typ"] for r in rows], y=[r["cells"][k]["celf"] for r in rows], name=f"CELF, k = {k}", marker_color=C.COLOR_A, opacity=0.5 if k == ks[0] else 0.9,
                             text=[de(r["cells"][k]["celf"], 0) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Gewinn-Auswertungen")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height=380, bottom=100)


def build_guarantee(result):
    """Verhältnis Greedy / Optimum aller kleinen Netze (sortiert) mit der Garantie 1 − 1/e."""
    fig = go.Figure()
    for k, vals in result["ratios"].items():
        ys = sorted(vals)
        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, mode="lines+markers", name=f"k = {k}", line=dict(width=2)))
    fig.add_hline(y=C.GUARANTEE, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), annotation_text="Garantie 1 − 1/e", annotation_position="bottom left")
    fig.update_xaxes(title_text="Netze, nach Verhältnis sortiert")
    fig.update_yaxes(title_text="Spread von Greedy / Optimum", range=[0.55, 1.03])
    return _base(fig, height=340, bottom=60)


def build_noise(rows):
    """Zahl verschiedener Starter-Mengen und Streuung des Spreads bei 50 und 300 Trainings-Schnappschüssen."""
    xs = [f"{r['train']} Schnappschüsse" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["distinct"] for r in rows], name="verschiedene Mengen (von 8 Läufen)", marker_color=C.COLOR_A, text=[de(r["distinct"], 0) for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=xs, y=[r["range"] for r in rows], name="Streuung des Spreads in Knoten", marker_color=C.COLOR_B, text=[de(r["range"], 1) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Zahl bzw. Knoten", rangemode="tozero")
    return _base(fig, height=320, bottom=60)
