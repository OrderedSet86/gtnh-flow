import pytest

from factory_graph import ProgramContext
from src.data.basicTypes import Ingredient as Ing
from src.data.basicTypes import IngredientCollection as IngCol
from src.data.basicTypes import Recipe
from src.gtnh.overclocks import OverclockHandler


@pytest.fixture
def overclock_handler():
    return OverclockHandler(ProgramContext('tests/sanity_config.yaml'))


@pytest.mark.parametrize(
    "coils",
    [
        "cupronickel",
        "Cupronickel",
        "hss-g",
        "HSS-G",
    ],
)
def test_coils_value_case_sensitivity(coils: str, overclock_handler: OverclockHandler):
    recipe = Recipe(
        "electric blast furnace",
        "ev",
        IngCol(Ing("galena dust", 9), Ing("oxygen gas", 27000)),
        IngCol(Ing("roasted lead dust", 9), Ing("ashes", 1), Ing("so2", 9000)),
        120,
        54,
        coils=coils,
        heat=1200,
    )

    # tests pass if no exceptions are thrown
    overclock_handler.overclockRecipe(recipe)


@pytest.mark.parametrize(
    "saw",
    [
        "saw",
        "Saw",
        "buzzsaw",
        "BuzzSaw",
    ],
)
def test_saw_value_case_sensitivity(saw: str, overclock_handler):
    recipe = Recipe(
        "tree growth simulator",
        "lv",
        IngCol(),
        IngCol(Ing("oak wood", 5)),
        32,
        5,
        saw_type=saw,
    )

    # tests pass if no exceptions are thrown
    overclock_handler.overclockRecipe(recipe)
