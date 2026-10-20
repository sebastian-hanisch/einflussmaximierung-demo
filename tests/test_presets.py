"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import ifm_constants as C
import ifm_evaluation as ev
import ifm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["netztyp"] in C.ALL_TYPES and C.N_MIN <= p["n"] <= C.N_MAX and C.C_MIN <= p["c"] <= C.C_MAX, name
        assert 0 <= p["seed"] <= C.SEED_MAX and C.STRENGTH_MIN <= p["strength"] <= C.STRENGTH_MAX and C.K_MIN <= p["k"] <= C.K_MAX and p["method"] in C.METHODS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET_NAME]) == ev.Settings()
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET_NAME


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_cover_the_teaching_cases():
    by_type = {p["netztyp"]: p for p in C.PRESETS.values()}
    assert {C.T_RANDOM, C.T_POWER, C.T_GRID, C.T_RING, C.T_HUBS, C.T_LINK} <= set(by_type)
    assert by_type[C.T_GRID]["method"] == C.M_CLOSENESS and by_type[C.T_RING]["method"] == C.M_DD_INDEX and by_type[C.T_HUBS]["method"] == C.M_CORE


def test_every_preset_runs_and_the_grid_preset_shows_the_closeness_failure():
    for name, p in C.PRESETS.items():
        an = ev.analyse(ev.Settings(**p))
        assert len(an.chosen) == an.k and an.spread(C.M_GREEDY) > 0, name
    grid = ev.analyse(ev.Settings(**C.PRESETS["Gitter: die Nähe versagt"]))
    assert grid.ratio(C.M_CLOSENESS) < 0.75 and grid.dist["method"] < 3.0 < grid.dist["greedy"]
