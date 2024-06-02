# Using PuLP dependency to seriously simplify setting up LP problem. It's just a wrapper to help set things up,
#  but it does come with a built-in solver (which we will also use)
from pulp import LpVariable, LpProblem, LpMinimize
import re
from collections import Counter

from ..graph import Graph
from ..graph._preProcessing import connectGraph, removeBackEdges
from ..data.basicTypes import Ingredient, IngredientCollection, Recipe
from factory_graph import ProgramContext

# Pulled from https://gtnh.miraheze.org/wiki/Commonly_used_acronyms_and_nicknames
common_acronyms = {}


class LPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.variables = []
        self.problem = LpProblem("Recipe Optimization", LpMinimize)
        self.solved_vars = None  # Result from linear solver
        self.lookup = {}
        self.ing_priorities = {}

    def genSlug(self, s):
        slug = re.sub(r"[^a-zA-Z0-9_]", "", re.sub(r"\W+", "_", s.lower()))
        if len(slug) > 25:
            return "".join(word[0] for word in slug.split("_"))
        else:
            return slug

    def genRecipeNames(self):
        node_recipes = {
            node: self.graph.recipes[node]
            for node in self.graph.nodes
            if node not in ["source", "sink"]
        }
        counter = Counter()
        name_map = {"sink": "sink", "source": "source"}
        for node, recipe in node_recipes.items():
            name = self.genSlug(recipe.machine + "_" + recipe.I[0].name)
            if name in counter:
                counter[name] += 1
                name += f"_{counter[name]}"
            name_map[node] = name
        return name_map

    # Create variables with somewhat nice names for debugging
    # Variables include all recipes and all items that are ever ingredients
    # Probably most inputs are net-zero (one output is just another recipe's input) and don't need a net input
    # However, it takes math to know ahead of time which those will be
    # (i.e. if an item is input and later output but in a lesser amount)
    # Since these LP variables strictly non-negative, they will only be inputs and will generally be optimized to zero.
    def createVariables(self):
        recipe_name_map = self.genRecipeNames()
        for node in self.graph.nodes:
            self.lookup[node] = LpVariable("rec_"+recipe_name_map[node], 0)
            self.variables.append(self.lookup[node])

        for ing in self.ing_priorities:
            self.lookup[ing] = LpVariable(self.genSlug(ing), 0)
            self.variables.append(self.lookup[ing])

    def genIngredientPriorities(self):
        ing_priorities = {}
        for recipe in self.graph.recipes.values():
            if hasattr(recipe, "cost_priority"):
                ing_priorities.update(
                    {ing: priority for ing, priority in recipe.cost_priority.items()}
                )
        if len(ing_priorities) == 0:
            raise RuntimeError(
                "No cost priorities found in recipes - needed for LP solver"
            )
        distinct_priorities = sorted(list(set(ing_priorities.values())))
        # In this program, each priority level gets a power of ten in the cost function.
        # Doubles have roughly 15 decimal places of precision, so this is a rough guess at the max number of priorities
        if len(distinct_priorities) >= 14:
            return RuntimeError(
                "Too many distinct priorities for LP solver, probably. Try making variables share priorities"
            )
        # +1 because 0 priority is special case for recipe tax
        priority_map = {old: new + 1 for new, old in enumerate(distinct_priorities)}
        ing_priorities = {
            ing: priority_map[priority] for ing, priority in ing_priorities.items()
        }
        return ing_priorities

    # Objective function in this case is cost, including recipe tax and prioritized cost variables
    def createObjectiveFunction(self):
        pass

    def run(self):
        # Map from ingredient name to priority level (normalized to 1,2, ...., n) for cost function
        self.ing_priorities = self.genIngredientPriorities()
        self.createVariables()
        # self.createConstraints()
        self.createObjectiveFunction()
        # self.solve()


def graphPreProcessing(self):
    connectGraph(self)
    # if not self.graph_config.get('KEEP_BACK_EDGES', False):
    #     removeBackEdges(self)
    Graph.createAdjacencyList(self)


def linearProgrammingSolver(
    self: ProgramContext, project_name: str, recipes: list[Recipe], graph_config: dict
):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    self._graph = g  # For test access
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
