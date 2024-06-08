from pulp import LpVariable, LpProblem, LpMinimize
import re
from collections import Counter

from ..graph import Graph
from ..graph._preProcessing import connectGraph, removeBackEdges
from ..data.basicTypes import Ingredient, IngredientCollection, Recipe
from factory_graph import ProgramContext


class LPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.variables = []
        self.problem = LpProblem("gtnh_flow_lp_solver", LpMinimize)
        self.solved_vars = None
        self.project = None
        
        # Solver Variables
        self.recipe_vars = None
        self.explicit_input_amounts = None
        self.additional_input_subtractors = None
        self.additional_input_switches = None
        self.byproduct_amounts = None

    # Create variables with somewhat nice names for debugging
    # Variables include:
    #   - All Recipes (these are the big ones we return)
    #   - Explicit inputs (inputs the user wants to provide)
    #   - Additional inputs + switches (ingredients that might need to be added to reach a solution)
    #   - Surplus Variables (item production beyond target is usually heavily penalized, but sometimes desired)
    def createVariables(self, project):
        recipe_vars = LpVariable.dicts("recipe", project.recipe_names, 0)
        explicit_input_amounts = LpVariable.dicts("input", project.explicit_inputs, 0)
        additional_input_subtractors = LpVariable.dicts("in_sub", additional_inputs, 0, 1)
        additional_input_switches = LpVariable.dicts("in_switch", additional_inputs, 0, 1, cat=LpBinary)
        byproduct_amounts = LpVariable.dicts("byproduct", project.variables, 0)


    # Objective function in this case is cost, including recipe tax and prioritized cost variables
    def createObjectiveFunction(self):
        pass

    def run(self):
        # Map from ingredient name to priority level (normalized to 1,2, ...., n) for cost function
        self.project = LpProject.fromGraph(self.graph)
        self.createVariables()
        # self.createConstraints()
        self.createObjectiveFunction()
        # self.solve()


def graphPreProcessing(self):
    connectGraph(self)
    # if not self.graph_config.get('KEEP_BACK_EDGES', False):
    #     removeBackEdges(self)
    # Graph.createAdjacencyList(self)


def linearProgrammingSolver(
    self: ProgramContext, project_name: str, recipes: list[Recipe], graph_config: dict
):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    self._graph = g  # type: ignore # For test access
    graphPreProcessing(g)

    lp = LPSolver(g)
    lp.run()


if __name__ == "__main__":
    # Usage (bc relative imports):
    # python -m src.graph._lpSolver
    c = ProgramContext()

    # c.run(graph_gen=linearProgrammingSolver)
    c.graph_gen = linearProgrammingSolver
    c.generate_one("power/fish/methane.yaml")
