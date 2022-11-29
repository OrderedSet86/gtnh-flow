import math

import pytest
import yaml

from dataClasses.load import recipesFromConfig
from graphClasses.graph import Graph

import json
def loadTestConfig():
    with open('config_factory_graph.yaml', 'r') as f:
        graph_config = yaml.safe_load(f)
    return graph_config


def test_balanceSimple():
    project_name = 'simpleGraph'

    # Load recipes
    recipes = recipesFromConfig(project_name, project_folder='tests/testProjects')

    # Create graph
    g = Graph(project_name, recipes, loadTestConfig())
    g.connectGraph()
    g.balanceGraph()

    expected_edge_and_quants = {
        ('source', '1', 'sugar beet'): 0.16,
        ('1', '0', 'sugar'): 1.28,
        ('0', 'sink', 'carbon dust'): 0.08,
        ('0', 'sink', 'oxygen'): 1000.0,
        ('0', 'sink', 'water'): 200.0,
    }

    for edge, quant in expected_edge_and_quants.items():
        assert edge in g.edges
        assert math.isclose(g.edges[edge]['quant'], quant, rel_tol=1e3)


def test_balanceLoop():
    # This is expected to have the same fail/pass as test_balanceSimple thanks to
    # g.removeBackEdges(), but adding this just in case.

    project_name = 'loopGraph'

    # Load recipes
    recipes = recipesFromConfig(project_name, project_folder='tests/testProjects')

    # Create graph
    g = Graph(project_name, recipes, loadTestConfig())
    g.connectGraph()
    g.balanceGraph()

    expected_edge_and_quants = {
        ('source', '1', 'acetic acid'): 100.0,
        ('source', '1', 'sulfuric acid'): 100.0,
        ('1', '0', 'diluted sulfuric acid'): 100.0,
        ('1', 'sink', 'ethenone'): 100.0,
        ('0', 'sink', 'sulfuric acid'): 66.67,
        ('0', 'sink', 'water'): 33.33,
    }

    for edge, quant in expected_edge_and_quants.items():
        assert edge in g.edges
        assert math.isclose(g.edges[edge]['quant'], quant, rel_tol=1e3)


def test_undeterminedMultiInput():
    # FIXME:
    assert 1 == 1