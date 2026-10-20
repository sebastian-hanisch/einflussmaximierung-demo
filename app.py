"""Einflussmaximierung: wen informiert man zuerst? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Reihe „Netzwerkanalyse“: mit k Startern eine Nachricht im Netz verbreiten (Independent Cascade), Greedy (mit CELF) gegen einfache Auswahlregeln (Grad, Nähe, Degree Discount ...),
dazu der Abstand der Starter, die Garantie 1 − 1/e gegen die Wirklichkeit und die Frage, wie eindeutig die Auswahl überhaupt ist.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ifm_constants as C
from ifm_evaluation import (
    Settings,
    analyse,
    celf_rows,
    curve_rows,
    de,
    distance_rows,
    guarantee_rows,
    noise_rows,
    verdict_celf,
    verdict_curves,
    verdict_distance,
    verdict_guarantee,
    verdict_main,
    verdict_noise,
)
from ifm_layout import dunkles_design, kopf_css
from ifm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ifm_scenario import link_stand
from ifm_visualization import (
    build_celf,
    build_curve,
    build_distance,
    build_guarantee,
    build_methods,
    build_network,
    build_noise,
    build_ratio_bars,
    build_redundancy,
)

st.set_page_config(page_title="Einflussmaximierung – Sebastian Hanisch", layout="wide")

BAR_K = 10   # Experiment 1: Verhältnis je Netztyp bei dieser Zahl von Startern


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _curves(n, c, strength):
    return curve_rows(n, c, strength)


@st.cache_data(show_spinner=False)
def _distance(n, c, strength, k):
    return distance_rows(n, c, strength, k)


@st.cache_data(show_spinner=False)
def _celf(n, c, strength):
    return celf_rows(n, c, strength)


@st.cache_data(show_spinner=False)
def _guarantee():
    return guarantee_rows()


@st.cache_data(show_spinner=False)
def _noise(typ, n, c, strength, k):
    return noise_rows(typ, n, c, strength, k)


def experiment(name, label, key, compute, show, hint):
    """Teure Zusatzansicht erst auf Klick; das Ergebnis bleibt nur für genau diese Einstellungen stehen (Session State)."""
    state_key = f"exp_{name}"
    if st.button(label, key=f"run_{name}", help=hint):
        with st.spinner("Rechne …"):
            result = compute()
        st.session_state[state_key] = (key, result)
    state = st.session_state.get(state_key)
    if state is None:
        st.caption(f"Noch nicht gerechnet. {hint}")
        return
    if state[0] != key:
        st.info("Die Einstellungen haben sich seit der letzten Rechnung geändert. Bitte neu starten.")
        return
    show(state[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ifm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📣 Einflussmaximierung: wen informiert man zuerst?")
st.markdown(
    """
**Fünftes Stück** der Reihe „Netzwerkanalyse“, im Ausbreitungs-Arm der Auftakt: In einem Netz von Subunternehmern und Lieferanten soll eine Störmeldung oder ein Verfahrenswechsel möglichst viele Betriebe erreichen, aber nur *k* Betriebe werden direkt informiert, und jeder Informierte gibt die Nachricht an jeden Partner nur mit einer bestimmten Wahrscheinlichkeit weiter (**Independent Cascade**). Wen wählt man? Die Demo stellt das **Greedy-Verfahren** (Kempe, Kleinberg, Tardos; mit der Beschleunigung CELF) den einfachen Regeln gegenüber (Grad, Nähe, Degree Discount …). Das Ausbreitungsmodell kennen Sie aus der [Demo zu Kaskaden](https://sebastianhanisch-kaskaden-demo.streamlit.app/), die Kernzahl aus der [k-Kern-Demo](https://sebastianhanisch-k-core-demo.streamlit.app/). Anders als die Fall-Demos zeigt diese Demo **ein** Problem an einem wachsenden Beispiel.

Die Kernfrage: **Wie viel verliert die einfache Regel gegenüber Greedy, und wo?** Der Hook ist der Schritt von einem zu mehreren Startern: wer die Starter einzeln nach ihrer Wirkung auswählt, wählt Knoten mit überlappender Reichweite.

Weiter unten: fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Greedy", expanded=True):
    st.markdown(
        """
1. **Independent Cascade:** jeder Informierte gibt die Nachricht an jeden Partner mit Wahrscheinlichkeit *p* weiter, jede Kante bekommt genau eine Chance. Das ist gleichbedeutend damit, dass jede Kante mit Wahrscheinlichkeit *p* „bleibt“ und alle Starter ihre Zusammenhangskomponente erreichen: ein **Schnappschuss** des Netzes.
2. **Ausbreitungsstärke:** statt *p* stellen Sie ein, wie viel Prozent der Betriebe ein zufälliger **einzelner** Starter im Mittel erreicht; die Demo rechnet daraus *p* aus (Bisektion) und zeigt es an.
3. **Spread:** die erwartete Zahl erreichter Betriebe, gemittelt über viele Schnappschüsse. **Greedy** rechnet auf 300 festen Trainings-Schnappschüssen; **bewertet** wird jede Auswahl auf 600 davon unabhängigen.
4. **Greedy:** wählt nacheinander den Knoten mit dem größten **Zuwachs** (noch nicht erreichte Betriebe). Weil der Zuwachs mit wachsender Menge nur fällt (**Submodularität**), erreicht Greedy mindestens den Anteil 1 − 1/e ≈ 63 % des Optimums.
5. **CELF** (lazy greedy) liefert dieselbe Auswahl mit viel weniger Rechnung: ein alter Zuwachs ist eine obere Schranke, nur der oberste Eintrag wird neu gerechnet.
6. **Regeln:** *Grad*, *Nähe*, *Betweenness*, *PageRank*, *Kernzahl* wählen die *k* besten Knoten nach dem Maß; **Degree Discount** (Chen, Wang, Yang) wertet nach jeder Wahl die Nachbarn ab, **Single Discount** zieht jedem Nachbarn einen Kontakt ab. Gleichstände werden zufällig aufgelöst und über zehn Ziehungen gemittelt; die Regel „kleinster Index“ steht eigens daneben.
7. **Abstand und Redundanz:** der mittlere Abstand der Starter im Netz, und der Spread der Menge geteilt durch die Summe der Einzelspreads (1 = keine Überlappung).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Netz zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    typ = st.selectbox("Netztyp", C.ALL_TYPES, key="netztyp_select", help="Welches Netz erzeugt oder geladen wird. Das Link-Netz ist fest.")
    fixed = typ in C.FIXED_TYPES
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", disabled=fixed, help="Zahl der Knoten (beim Gitter auf ein Quadrat gerundet).")
    c = st.slider("Verbindungen je Knoten c", *bounds("c_slider"), key="c_slider", disabled=fixed, help="Was c bedeutet, hängt vom Netztyp ab (siehe darunter).")
    st.caption(C.C_MEANING[typ] if typ in C.C_MEANING else "Dieses Netz ist fest: n, c und das Netz selbst hängen nicht vom Zufalls-Seed ab (der Seed bestimmt weiter die Schnappschüsse).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt das Netz und die Schnappschüsse.")
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Ausbreitung und Auswahl**")
    strength = st.slider("Ausbreitungsstärke (%)", *bounds("strength_slider"), key="strength_slider",
                         help="Anteil der Betriebe, den ein zufälliger einzelner Starter im Mittel erreicht. Daraus wird die Weitergabewahrscheinlichkeit p berechnet.")
    k = st.slider("Zahl der Starter k", *bounds("k_slider"), key="k_slider", help="Wie viele Betriebe direkt informiert werden.")
    method = st.selectbox("Verfahren", C.METHODS, key="method_select", help="Welche Auswahl im Netz gezeigt wird; die Spreads aller Verfahren stehen im Diagramm darunter.")
    st.caption(C.METHOD_HELP[method])

sync_query_params({"netztyp_select": typ, "n_slider": int(n), "c_slider": int(c), "seed_input": int(seed), "strength_slider": int(strength), "k_slider": int(k), "method_select": method})

settings = Settings(typ, int(n), int(c), int(seed), int(strength), int(k), method)
with st.spinner("Rechne Schnappschüsse und Auswahlen …"):
    an = _analysis(settings)
if typ == C.T_LINK:
    st.info(f"Das echte Link-Netz von sebastianhanisch.net, Schnappschuss vom {link_stand()} (ungerichtet zusammengefasst): 50 Seiten und 300 Demos. Die Zahlen gelten für diesen Stand, nicht für die heutige Website.")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Starter und ihre Reichweite")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten / Kanten", f"{an.inst.n} / {an.inst.m}")
m2.metric("Weitergabewahrscheinlichkeit p", de(an.prep.p, 3), delta=f"Ausbreitungsstärke {settings.strength} %", delta_color="off")
m3.metric(f"Spread von Greedy (k = {an.k})", de(an.spread(C.M_GREEDY), 1), delta=f"± {de(an.se_greedy, 1)} (Standardfehler)", delta_color="off")
m4.metric(f"Spread „{an.method}“", de(an.spread(an.method), 1), delta=f"{de(100 * an.ratio(an.method), 0)} % von Greedy", delta_color="off")
st.markdown(f"**{an.method}: die {an.k} Starter** (rot umrandet, Nummer = Wahlreihenfolge; Farbe der Knoten = Wahrscheinlichkeit, von den Startern erreicht zu werden)")
st.plotly_chart(build_network(an.inst, an.reach, an.chosen, an.method), width="stretch", key="net_chart")
st.markdown("**Alle Verfahren im Vergleich** (derselbe Satz Schnappschüsse; Regeln mit Gleichständen als Mittel über zehn Auflösungen)")
st.plotly_chart(build_methods(an), width="stretch", key="methods_chart")
st.caption(f"Mittlerer Abstand der Starter: {de(an.dist['method'], 1)} (Greedy: {de(an.dist['greedy'], 1)}), Redundanz {de(an.redund['method'], 2)} (Greedy: {de(an.redund['greedy'], 2)}). "
           f"Greedy braucht {an.naive_evals} Gewinn-Auswertungen, CELF {an.celf_evals} bei gleicher Auswahl.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel verliert die einfache Regel gegenüber Greedy?")
kind, text = verdict_main(an)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

n_exp, c_exp, s_exp, k_exp = settings.n, settings.c, settings.strength, settings.k
curve_type = typ if typ in C.GENERATED_TYPES else C.T_RANDOM
st.subheader("🔬 Experiment 1: Wo liegen die einfachen Regeln nahe bei Greedy?")
st.caption(f"Spread geteilt durch den Spread von Greedy, bewertet auf unabhängigen Schnappschüssen; Median über {len(C.LIVE_SEEDS)} feste Netze je Netztyp, n = {n_exp}, c = {c_exp}, Ausbreitungsstärke {s_exp} %. "
           "Verhältnisse knapp über 1 sind Schätzrauschen (Greedy ist auf den Trainings-Schnappschüssen optimiert, nicht auf den Bewertungs-Schnappschüssen).")


def show_curves(rows):
    row = next(r for r in rows if r["typ"] == curve_type)
    st.markdown(f"**Verlauf über k in „{curve_type}“**")
    st.plotly_chart(build_curve(row), width="stretch", key="curve_chart")
    st.markdown(f"**Alle Netztypen bei k = {BAR_K}**")
    st.plotly_chart(build_ratio_bars(rows, BAR_K), width="stretch", key="ratio_chart")
    kind_, text_ = verdict_curves(rows, BAR_K)
    getattr(st, kind_)(text_)


experiment("curves", "Regeln vergleichen", (n_exp, c_exp, s_exp), lambda: _curves(n_exp, c_exp, s_exp), show_curves, "Rechnet sieben Netztypen mit je vier Netzen und allen Verfahren, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Wählt Greedy Starter, die weiter auseinander liegen?")
st.caption(f"Mittlerer Abstand der k = {k_exp} Starter und ihre Redundanz (Spread der Menge geteilt durch die Summe der Einzelspreads) für Greedy, Grad und Nähe; n = {n_exp}, c = {c_exp}, Ausbreitungsstärke {s_exp} %.")


def show_distance(rows):
    st.plotly_chart(build_distance(rows), width="stretch", key="distance_chart")
    st.plotly_chart(build_redundancy(rows), width="stretch", key="redundancy_chart")
    kind_, text_ = verdict_distance(rows)
    getattr(st, kind_)(text_)


experiment("distance", "Abstände rechnen", (n_exp, c_exp, s_exp, k_exp), lambda: _distance(n_exp, c_exp, s_exp, k_exp), show_distance, "Rechnet sieben Netztypen mit je vier Netzen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Wie viel spart CELF?")
st.caption(f"Zahl der Gewinn-Auswertungen für k = {C.CELF_KS[0]} und k = {C.CELF_KS[1]}: naives Greedy gegen CELF, bei gleicher Auswahl; n = {n_exp}, c = {c_exp}, Ausbreitungsstärke {s_exp} %.")


def show_celf(rows):
    st.plotly_chart(build_celf(rows), width="stretch", key="celf_chart")
    kind_, text_ = verdict_celf(rows)
    getattr(st, kind_)(text_)


experiment("celf", "CELF vergleichen", (n_exp, c_exp, s_exp), lambda: _celf(n_exp, c_exp, s_exp), show_celf, "Rechnet sieben Netztypen mit je vier Netzen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Garantie gegen Wirklichkeit")
st.caption(f"{C.SMALL_NETS} kleine Zufallsnetze (n = {C.SMALL_N[0]} bis {C.SMALL_N[1]}, {C.SMALL_M[0]} bis {C.SMALL_M[1]} Kanten, p = {de(C.SMALL_P, 1)}) mit **exaktem** Spread (Aufzählung aller Kantenzustände): Greedy gegen das Optimum durch Aufzählung aller Mengen aus k = {C.SMALL_K[0]} und {C.SMALL_K[1]} Knoten.")


def show_guarantee(result):
    st.plotly_chart(build_guarantee(result), width="stretch", key="guarantee_chart")
    kind_, text_ = verdict_guarantee(result)
    getattr(st, kind_)(text_)


experiment("guarantee", "Garantie prüfen", "fest", _guarantee, show_guarantee, "Rechnet 40 Netze mit je 256 bis 4 096 Zuständen, wenige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 5: Ist die Auswahl von Greedy eindeutig?")
st.caption(f"Greedy läuft {C.NOISE_RUNS}-mal auf frischen Schnappschüssen (50 und 300 Stück) im gewählten Netztyp („{typ}“), k = {k_exp}; jede Menge wird auf denselben unabhängigen Schnappschüssen bewertet.")


def show_noise(rows):
    st.plotly_chart(build_noise(rows), width="stretch", key="noise_chart")
    kind_, text_ = verdict_noise(rows)
    getattr(st, kind_)(text_)


experiment("noise", "Läufe wiederholen", (typ, n_exp, c_exp, s_exp, k_exp), lambda: _noise(typ, n_exp, c_exp, s_exp, k_exp), show_noise, "Rechnet 64 Greedy-Läufe auf vier Netzen, einige Sekunden.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleiche Weitergabewahrscheinlichkeit, ungerichtet** | Jede Kante hat dasselbe *p*. Mit Gewichten, Richtungen oder einem Schwellenmodell (Linear Threshold) sind Spread und Garantie andere Größen; die Lücken der Regeln verschieben sich. | Stück 6 (komplexe Ansteckung) |
| **Kalibrierung der Ausbreitungsstärke** | *p* wird über einen zufälligen **Einzelstarter** festgelegt. Eine andere Kalibrierung (etwa über das Maximum oder über mehrere Starter) ergibt andere Lücken. | – |
| **Schätzung mit Schnappschüssen** | Der Spread ist ein Mittel über 300 bzw. 600 Schnappschüsse; der Standardfehler liegt bei einem bis zwei Knoten. Verhältnisse knapp über 1 sind Rauschen: Greedy ist auf den Trainings-Schnappschüssen optimiert. | – |
| **Eindeutigkeit der Auswahl** | Greedy liefert eine Menge, nicht *die* beste: auf frischen Schnappschüssen entstehen andere Mengen mit fast gleichem Spread (Experiment 5). | – |
| **Gleichstände** | Auf regelmäßigen Netzen (Gitter, Ring) haben viele Knoten denselben Wert. Die Regeln lösen sie zufällig auf (Mittel über zehn Ziehungen); mit „kleinste Knotennummer“ verliert Degree Discount dort bis zu 26 Prozentpunkte. | – |
| **Keine Kosten je Betrieb** | Alle Starter kosten gleich viel. Mit Kosten gilt die Garantie von Greedy nicht mehr in dieser Form (Leskovec u. a. 2007 geben eine Variante an). | – |
| **Synthetische Netze** | Die Zahlen gelten für diese Netztypen bei diesen Größen (Median über feste Netze). Das Link-Netz ist ein Sonderfall: die Startseite ist mit allen Seiten verlinkt, ein Starter erreicht fast alles. | Stück 7 (Eindämmung) |
| **Stand der Technik** | Für große Netze rechnet man heute mit Reverse-Influence-Sampling (TIM+, IMM) statt mit Schnappschüssen und CELF; die Demo zeigt die Lehrfassung. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Independent Cascade.** Ein Knoten, der in Runde $t$ informiert wird, bekommt einmal die Chance, jeden noch nicht informierten Nachbarn mit Wahrscheinlichkeit $p$ zu informieren. Für ungerichtete Netze mit gleichem $p$ ist das gleichbedeutend mit Erreichbarkeit im Zufallsnetz $G_p$, in dem jede Kante unabhängig mit Wahrscheinlichkeit $p$ bleibt (Schnappschuss). Der **erwartete Spread** einer Startmenge $S$ ist

$$\sigma(S) = \mathbb{E}\bigl[\,|R_{G_p}(S)|\,\bigr]$$

wobei $R_{G_p}(S)$ die Menge der von $S$ in $G_p$ erreichbaren Knoten ist. Mit $R$ Schnappschüssen $G^{(1)}, \dots, G^{(R)}$ wird er geschätzt durch

$$\hat\sigma(S) = \frac{1}{R} \sum_{r=1}^{R} \bigl|R_{G^{(r)}}(S)\bigr|$$

**Zuwachs** eines Knotens $v$ zur Menge $S$:

$$\Delta(v \mid S) = \sigma(S \cup \{v\}) - \sigma(S)$$

**Submodularität und Garantie** (Kempe, Kleinberg, Tardos 2003): $\sigma$ ist monoton und submodular, also gilt für $S \subseteq T$

$$\Delta(v \mid S) \ge \Delta(v \mid T)$$

und das Greedy-Ergebnis $S_\mathrm{G}$ mit $k$ Startern erreicht

$$\sigma(S_\mathrm{G}) \ge \Bigl(1 - \frac{1}{e}\Bigr) \sigma(S^*)$$

gegenüber dem Optimum $S^*$.

**Degree Discount** (Chen, Wang, Yang 2009): nach der Wahl von $t_v$ Nachbarn des Knotens $v$ mit Grad $d_v$ ist sein abgewerteter Grad

$$dd_v = d_v - 2\, t_v - (d_v - t_v)\, t_v\, p$$

**Redundanz** einer Menge $S$ (1 = die Reichweiten überlappen nicht):

$$\rho(S) = \frac{\sigma(S)}{\sum_{v \in S} \sigma(\{v\})}$$

**Literatur.**
- Kempe, D., Kleinberg, J., Tardos, É. (2003): *Maximizing the spread of influence through a social network.* KDD 2003, 137–146.
- Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., Glance, N. (2007): *Cost-effective outbreak detection in networks.* KDD 2007. (CELF)
- Chen, W., Wang, Y., Yang, S. (2009): *Efficient influence maximization in social networks.* KDD 2009. (NewGreedy, Degree Discount)
- Kitsak, M. u. a. (2010): *Identification of influential spreaders in complex networks.* Nature Physics 6, 888–893. (Kernzahl und Abstand der Starter)
- Tang, Y., Xiao, X., Shi, Y. (2014): *Influence maximization: near-optimal time complexity meets practical efficiency.* SIGMOD 2014. (TIM/TIM+, Reverse-Influence-Sampling)
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
