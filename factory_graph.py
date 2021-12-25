# Standard libraries
import json
import sys
from collections import defaultdict, deque, OrderedDict
from pathlib import Path

# Pypi libraries
from jsmin import jsmin
from termcolor import cprint

# Internal libraries
from graphClasses.graph import Graph
from dataClasses.base import Ingredient, IngredientCollection, Recipe
from dataClasses.load import recipesFromConfig


if __name__ == '__main__':
    graph_config = {
        # TODO: Add these to backend
        'PRECISION': 4,
        'ROUND_MACHINES_UP': False,
        'POWER_LINE': True,
    }

    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        raise RuntimeError('Need a project config name as an argument! Will be pulled from projects/{name}.json')

    recipes = recipesFromConfig(project_name)

    for rec in recipes:
        print(rec)

    # Create graph and render
    g = Graph(project_name, recipes, graph_config)
    g.connectGraph()
    g.outputGraphviz()