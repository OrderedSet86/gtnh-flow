import pytest

from src.data.basicTypes import Ingredient
from src.data.basicTypes import IngredientCollection


oak_wood_input = Ingredient("oak wood", 16)
nitrogen_input = Ingredient("nitrogen", 1000)
oak_wood_oven_recipe = IngredientCollection(oak_wood_input, nitrogen_input)


@pytest.mark.parametrize(
    "collection,item,expected",
    [
        (oak_wood_oven_recipe, "oak wood", True),
        (oak_wood_oven_recipe, "nitrogen", True),
        (oak_wood_oven_recipe, nitrogen_input, True),
        (oak_wood_oven_recipe, oak_wood_input, True),
        (oak_wood_oven_recipe, Ingredient("oak wood", 16), True),
        (oak_wood_oven_recipe, Ingredient("oak wood", 1), True),
        (oak_wood_oven_recipe, Ingredient("oak wood", 0), False),
        (oak_wood_oven_recipe, Ingredient("oak wood", 20), False),
        # Case sensitive
        (oak_wood_oven_recipe, "Nitrogen", False),
        (oak_wood_oven_recipe, "Oak Wood", False),
        (oak_wood_oven_recipe, Ingredient("Oak Wood", 15), False),
    ],
)
def test_IngredientCollection_contains(collection, item, expected):
    assert expected == (item in collection)
