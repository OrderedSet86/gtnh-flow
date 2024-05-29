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
    # for now it uses config file in the project
    # it may cause issue if config file is messed up
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
    

recipe_volcanus = mod_recipe(recipe_ebf, machine="volcanus")
recipe_samarium = Recipe(
    "volcanus",
    "ev",
    IngCol(
        Ing("Dephosphated Samarium Concentrate Dust", 1),
    ),
    IngCol(
        Ing("Samarium Oxide Dust", 1),
        Ing("Gangue Dust", 1),
    ),
    514,
    2,
    coils="HSS-G",
    heat=1200,
)

# Based on in-game measurements
@pytest.mark.parametrize(
    "recipe,expected_eut,expected_dur,expected_parallel",
    [
        (
        # No Overclocks (800K over recipe - no bonuses)
            mod_recipe(recipe_volcanus, user_voltage="mv", coils="cupronickel"),  # 1801K    
            120 * 0.9,
            25 / 2.2,
            1,
        ),
        (
        # No Overclocks (1700K over recipe - one 5% heat bonus)
            mod_recipe(recipe_volcanus, user_voltage="mv", coils="kanthal"),  # 2701K    
            120 * 0.9 * 0.95,
            25 / 2.2,
            1,
        ),
        # 4x voltage for volcanus is 4x parallels
        (
            mod_recipe(recipe_volcanus, user_voltage="hv", coils="cupronickel"),  # 1801K    
            120 * 4 * 0.9,
            25 / 2.2,
            4,
        ),
        # EBF heat bonuses are applied after parallels are calculated (so still only 4 parallels)
        (
            mod_recipe(recipe_volcanus, user_voltage="hv", coils="HSS-G"),  # 5401K
            120 * 4 * 0.9 * 0.95**4,
            25 / 2.2,
            4
        ),
        # EV is enough for 16x parallels but capped to 8x. Not enough for overclock.
        (
            mod_recipe(recipe_volcanus, user_voltage="ev", coils="cupronickel"),  # 1801K
            120 * 8 * 0.9,
            25 / 2.2,
            8,
        ),
        # IV is enough for 8 parallels and 1 normal overclock.
        (
            mod_recipe(recipe_volcanus, user_voltage="iv", coils="cupronickel"),  # 1801K
            120 * 8 * 4 * 0.9,
            25 / 2.2 / 2,
            8,
        ),
        # 8 Parallel, 1 perfect oc (two 5% heat eut bonuses)
        (
            mod_recipe(recipe_volcanus, user_voltage="iv", coils="nichrome"),  # 3601K
            120 * 8 * 4 * 0.9 * 0.95**2,
            25 / 2.2 / 4,
            8,
        ),
        # 8 Parallel, 1 perfect oc, 1 normal oc (two 5% heat eut bonuses)
        (
            mod_recipe(recipe_volcanus, user_voltage="luv", coils="nichrome"),  # 3601K
            120 * 8 * 4 * 4 * 0.9 * 0.95**2,
            25 / 2.2 / 4 / 2,
            8,
        ),
        # Recipe shows gtpp eut discount applied before parallels (would be 3 parallels otherwise)
        (
            recipe_samarium,  # 3601K
            514 * 4 * 0.9 * 0.95**4,
            2 / 2.2,
            4,
        ),
    ],
)
def test_volcanusOverclock(recipe, expected_eut, expected_dur, expected_parallel, overclock_handler):
    overclocked = overclock_handler.overclockRecipe(recipe)
    assert overclocked.eut == expected_eut
    assert overclocked.dur == expected_dur
    assert overclocked.parallel == expected_parallel


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
