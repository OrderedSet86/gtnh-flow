import json
from collections import OrderedDict
from pathlib import Path

from jsmin import jsmin

from dataClasses.base import Ingredient, IngredientCollection, Recipe


def recipesFromConfig(project_name, project_folder='projects'):
    # Load config file
    CONFIG_FILE_PATH = Path(project_folder) / f'{project_name}.json'
    project_name = CONFIG_FILE_PATH.name.split('.')[0]
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = json.loads(jsmin(f.read()), object_pairs_hook=OrderedDict)

    # FIXME: Hack to get access to config_factory_graph.jsonc
    user_config_path = Path(__file__).absolute().parent.parent / 'config_factory_graph.jsonc'
    with open(user_config_path, 'r') as f:
        graph_config = json.loads(jsmin(f.read()))

    # Prep recipes for graph
    recipes = []
    for rec in config:
        if graph_config.get('DUR_FORMAT', 'ticks') == 'sec':
            rec['dur'] *= 20
        recipes.append(
            Recipe(
                rec['m'],
                rec['tier'],
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['I'].items()]),
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['O'].items()]),
                rec['eut'],
                rec['dur'],
                **{x: rec[x] for x in rec.keys() if x not in {'m', 'I', 'O', 'eut', 'dur', 'tier'}},
            )
        )

    return recipes