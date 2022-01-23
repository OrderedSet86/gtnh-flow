# Standard libraries
import argparse
import json
import sys

# Pypi libraries
from jsmin import jsmin

# Internal libraries
from graphClasses.graph import Graph
from dataClasses.load import recipesFromConfig


if __name__ == '__main__':
    with open('config_factory_graph.jsonc', 'r') as f:
        graph_config = json.loads(jsmin(f.read()))

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
