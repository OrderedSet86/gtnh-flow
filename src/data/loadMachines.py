from pathlib import Path

import yaml

from src.data.basicTypes import Ingredient, IngredientCollection, Recipe


def recipesFromConfig(project_name, project_folder='projects'):
    # Load config file
    CONFIG_FILE_PATH = Path(project_folder) / f'{project_name}'
    project_name = CONFIG_FILE_PATH.name.split('.')[0]
    with open(CONFIG_FILE_PATH, 'rb') as f:
        config = yaml.safe_load(f)

    user_config_path = Path(__file__).absolute().parent.parent.parent / 'config_factory_graph.yaml'
    with open(user_config_path, 'r') as f:
        graph_config = yaml.safe_load(f)

    # Prep recipes for graph
    recipes = []
    for rec in config:
        if graph_config.get('DUR_FORMAT', 'ticks') == 'sec':
            rec['dur'] *= 20

        machine_name = rec['m'].lower()

        recipes.append(
            Recipe(
                machine_name,
                rec['tier'].lower(),
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['I'].items()]),
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['O'].items()]),
                rec['eut'],
                rec['dur'],
                **{x: rec[x] for x in rec.keys() if x not in {'m', 'I', 'O', 'eut', 'dur', 'tier'}},
            )
        )

    return recipes
