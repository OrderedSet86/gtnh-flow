# Standard libraries
import argparse
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
        'POWER_LINE': True, # Automatically burns all leftover fuels
        'DO_NOT_BURN': {
            'methanol'
        },
        'DEBUG_SHOW_EVERY_STEP': False, # Outputs graphviz on each compute step
            # ^ (you will generally never want this to be true if you're a user)

        # TODO: Add below to backend
        'PRECISION': 4,
        'ROUND_MACHINES_UP': False,
    }

    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        raise RuntimeError('Need a project config name as an argument! Will be pulled from projects/{name}.json')

    recipes = recipesFromConfig(project_name)

    for rec in recipes:
        print(rec)
    print()

    # Create graph and render
    g = Graph(project_name, recipes, graph_config)
    g.connectGraph()
    g.balanceGraph()
    g.outputGraphviz()
