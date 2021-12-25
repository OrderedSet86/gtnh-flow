import pytest

from dataClasses.load import recipesFromConfig
from graphClasses.graph import Graph


# Note that recipe ordering is deterministic!
# (Thanks to the OrderedDict hook in dataClasses.load.recipesFromConfig)


def test_connectionSimple():
    project_name = 'simpleGraph'

    # Load recipes
    recipes = recipesFromConfig(project_name, project_folder='tests/testProjects')

    # Create graph
    g = Graph(project_name, recipes)
    g.connectGraph()

    ### Check connections
    # 0: electrolyzer
    # 1: extractor

    expected_edges = [
        ('source', '1', 'sugar beet'),
        ('1', '0', 'sugar'),
        ('0', 'sink', 'carbon dust'),
        ('0', 'sink', 'oxygen'),
        ('0', 'sink', 'water')
    ]

    assert set(expected_edges) == set(g.edges.keys())


def test_connectionLoop():
    # Loops won't cause problems until you try to balance
    pass