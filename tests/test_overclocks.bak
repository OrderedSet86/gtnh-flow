from copy import deepcopy

import pytest
import yaml

from dataClasses.base import Ingredient, Recipe, IngredientCollection
from gtnhClasses.overclocks import overclockRecipe

import json
def loadTestConfig():
    with open('config_factory_graph.yaml', 'r') as f:
        graph_config = yaml.safe_load(f)
    return graph_config


def test_standardOverclock():
    r_base = Recipe(
        'centrifuge',
        'MV',
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


def test_perfectOverclock():
    r_base = Recipe(
        'large chemical reactor',
        'MV',
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
    assert r.dur == 20

    r = deepcopy(r_base)
    r.user_voltage = 'HV'
    r = overclockRecipe(r)

    assert r.eut == 80
    assert r.dur == 5


def test_pyrolyseOverclock():
    r_base = Recipe(
        'pyrolyse oven',
        'MV',
        IngredientCollection(
            Ingredient('oak wood', 16),
            Ingredient('nitrogen', 1000),
        ),
        IngredientCollection(
            Ingredient('charcoal', 20),
            Ingredient('wood tar', 1500),
        ),
        96,
        320
    )

    r = deepcopy(r_base)
    r.user_voltage = 'HV'
    r.coils = 'kanthal'
    r = overclockRecipe(r)

    assert r.eut == 384
    assert r.dur == 160

    r = deepcopy(r_base)
    r.user_voltage = 'MV'
    r.coils = 'cupronickel'
    r = overclockRecipe(r)

    assert r.eut == 96
    assert r.dur == 640


def test_EBFOverclock():
    r_base = Recipe(
        'electric blast furnace',
        'LV',
        IngredientCollection(
            Ingredient('iron dust', 1),
            Ingredient('oxygen gas', 1000),
        ),
        IngredientCollection(
            Ingredient('steel ingot', 1),
            Ingredient('tiny pile of ashes', 1),
        ),
        120,
        500,
        heat=1000,
    )

    r = deepcopy(r_base)
    r.user_voltage = 'LV'
    r.coils = 'cupronickel'
    r = overclockRecipe(r)

    assert r.eut == 120
    assert r.dur == 500

    r = deepcopy(r_base)
    r.user_voltage = 'MV'
    r.coils = 'kanthal' # 2701K
    r = overclockRecipe(r)

    # excess heat = 1701K
    # should get 0.95x eut and one 2x OC

    assert r.eut == 120*4*.95
    assert r.dur == 500/2

    r = deepcopy(r_base)
    r.user_voltage = 'MV'
    r.coils = 'nichrome' # 3601K
    r = overclockRecipe(r)

    # excess heat = 2601K
    # should get (0.95**2)x eut and one 4x OC

    assert r.eut == 120*4*(.95**2)
    assert r.dur == 500/4

    r = deepcopy(r_base)
    r.user_voltage = 'HV'
    r.coils = 'nichrome' # 3601K
    r = overclockRecipe(r)

    # excess heat = 2601K
    # should get (0.95**2)x eut, one 4x OC, and one 2x OC

    assert r.eut == 120*(4**2)*(.95**2)
    assert r.dur == 500/4/2