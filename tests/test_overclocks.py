from copy import deepcopy

import pytest

from factory_graph import ProgramContext
from src.data.basicTypes import Ingredient as Ing
from src.data.basicTypes import IngredientCollection as IngCol
from src.data.basicTypes import Recipe
from src.gtnh.overclocks import OverclockHandler


def mod_recipe(recipe, **kwargs):
    modded = deepcopy(recipe)
    for k, v in kwargs.items():
        setattr(modded, k, v)
    return modded


@pytest.fixture
def overclock_handler():
    return OverclockHandler(ProgramContext())


recipe_sb_centrifuge = Recipe(
    "centrifuge",
    "lv",
    IngCol(Ing("glass dust", 1)),
    IngCol(Ing("silicon dioxide", 1)),
    5,
    80,
)


@pytest.mark.parametrize(
    "recipe,expected_eut,expected_dur",
    [
        (mod_recipe(recipe_sb_centrifuge, user_voltage="mv"), 20, 40),
        (mod_recipe(recipe_sb_centrifuge, user_voltage="hv"), 80, 20),
    ],
)
def test_standardOverclock(recipe, expected_eut, expected_dur, overclock_handler):
    overclocked = overclock_handler.overclockRecipe(recipe)
    assert overclocked.eut == expected_eut
    assert overclocked.dur == expected_dur


recipe_lcr = Recipe(
    "large chemical reactor",
    "lv",
    IngCol(Ing("glass dust", 1)),
    IngCol(Ing("silicon dioxide", 1)),
    5,
    80,
)


@pytest.mark.parametrize(
    "recipe,expected_eut,expected_dur",
    [
        (mod_recipe(recipe_lcr, user_voltage="mv"), 20, 20),
        (mod_recipe(recipe_lcr, user_voltage="hv"), 80, 5),
    ],
)
def test_perfectOverclock(recipe, expected_eut, expected_dur, overclock_handler):
    overclocked = overclock_handler.overclockRecipe(recipe)
    assert overclocked.eut == expected_eut
    assert overclocked.dur == expected_dur


recipe_ebf = Recipe(
    "electric blast furnace",
    "mv",
    IngCol(
        Ing("iron dust", 1),
        Ing("oxygen gas", 1000),
    ),
    IngCol(
        Ing("steel ingot", 1),
        Ing("tiny pile of ashes", 1),
    ),
    120,
    25,
    coils="cupronickel",
    heat=1000,
    circuit=11,
)


@pytest.mark.parametrize(
    "recipe,expected_eut,expected_dur",
    [
        # one normal OC
        (
            mod_recipe(recipe_ebf, user_voltage="hv", coils="kanthal"),  # 2701K
            120 * 4 * 0.95,
            25 / 2,
        ),
        # one perfect OC
        (
            mod_recipe(recipe_ebf, user_voltage="hv", coils="nichrome"),  # 3601K
            120 * 4 * 0.95**2,
            25 / 4,
        ),
        # one normal OC plus one perfect OC
        (
            mod_recipe(recipe_ebf, user_voltage="ev", coils="nichrome"),  # 3601K
            120 * 16 * 0.95**2,
            25 / 4 / 2,
        ),
    ],
)
def test_EBFOverclock(recipe, expected_eut, expected_dur, overclock_handler):
    overclocked = overclock_handler.overclockRecipe(recipe)
    assert overclocked.eut == expected_eut
    assert overclocked.dur == expected_dur


recipe_pyrolyse_oven = Recipe(
    "pyrolyse oven",
    "mv",
    IngCol(
        Ing("oak wood", 16),
        Ing("nitrogen", 1000),
    ),
    IngCol(
        Ing("charcoal", 20),
        Ing("wood tar", 1500),
    ),
    96,
    16,
    coils="cupronickel",
    circuit=10,
)


@pytest.mark.parametrize(
    "recipe,expected_eut,expected_dur",
    [
        # speed penalty
        (mod_recipe(recipe_pyrolyse_oven), 96, 16 / 0.5),
        # no penalty & no bonus
        (mod_recipe(recipe_pyrolyse_oven, coils="kanthal"), 96, 16),
        # speed bonus
        (mod_recipe(recipe_pyrolyse_oven, coils="nichrome"), 96, 16 / 1.5),
        # speed bonus & OC
        (
            mod_recipe(recipe_pyrolyse_oven, coils="nichrome", user_voltage="hv"),
            96 * 4,
            16 / 2 / 1.5,
        ),
    ],
)
def test_pyrolyseOverclock(recipe, expected_eut, expected_dur, overclock_handler):
    overclocked = overclock_handler.overclockRecipe(recipe)
    assert overclocked.eut == expected_eut
    assert overclocked.dur == expected_dur
