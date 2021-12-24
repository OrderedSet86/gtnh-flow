from copy import deepcopy

from ..dataClasses.recipe import Ingredient, Recipe, IngredientCollection
from ..gtnhClasses.overclocks import overclockRecipe


def test_standardOverclock():
    r_base = Recipe(
        'centrifuge',
        IngredientCollection(
            Ingredient('glass dust', 1)
        ),
        IngredientCollection(
            Ingredient('silicon dioxide', 1)
        ),
        5,
        80
    )
    r = deepcopy(r_base)
    r.user_voltage = 'MV'
    r = overclockRecipe(r)

    assert r.eut == 20
    assert r.dur == 40

    r = deepcopy(r_base)
    r.user_voltage = 'HV'
    r = overclockRecipe(r)

    assert r.eut == 80
    assert r.dur == 20
