# Standard libraries
import itertools
import json
import math
import sys
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

# Pypi libraries
import graphviz
from jsmin import jsmin
from ksuid import ksuid
from termcolor import cprint

# Internal libraries
from graphClasses.graph import Graph
from dataClasses.base import Ingredient, IngredientCollection, Recipe


if __name__ == '__main__':
    PRECISION = 4


    project_name = sys.argv[1]
    CONFIG_FILE_PATH = Path(f'projects/{project_name}.json')

    # Load config file
    project_name = CONFIG_FILE_PATH.name.split('.')[0]
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = json.loads(jsmin(f.read()), object_pairs_hook=OrderedDict)

    # Prep recipes for graph
    recipes = []
    for rec in config:
        recipes.append(
            Recipe(
                rec['m'],
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['I'].items()]),
                IngredientCollection(*[Ingredient(name, quant) for name, quant in rec['O'].items()]),
                rec['eut'],
                rec['dur'],
                **{x: rec[x] for x in vars(rec)}
            )
        )

    for rec in recipes:
        print(rec)

    # Create graph and render
    g = Graph(project_name, recipes)
    g.connectGraph()
    g.outputGraphviz()