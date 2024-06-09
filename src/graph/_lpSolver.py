import cvxpy as cp
import re
from collections import Counter

import numpy as np
from termcolor import colored

from ..graph import Graph
from ..graph._preProcessing import connectGraph, removeBackEdges
from ..data.basicTypes import Ingredient, IngredientCollection, Recipe
from factory_graph import ProgramContext
from ._lpProject import LpProject
from ._lpScaledMatrix import LpScaledMatrix


class LPSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.variables = []
        self.solved_vars = None

        # Solver Variables
        self.extra_inputs = []
        self.desired_byproducts = []
        self.minimized_byproducts = []
        self.recipe_vars, self.recipe_vars_dict = None, {}
        self.explicit_in_amounts, self.explicit_in_amounts_dict = None, {}
        self.extra_in_subs, self.extra_in_sub_dict = None, {}
        self.extra_in_switches, self.extra_in_switch_dict = None, {}
        self.surplus_amounts, self.surplus_amounts_dict = None, {}

    # Create variables with somewhat nice names for debugging
    # Variables include:
    #   - All Recipes (these are the big ones we return)
    #   - Explicit inputs (inputs the user wants to provide)
    #   - Additional inputs + switches (ingredients that might need to be added to reach a solution)
    #   - Surplus Variables (item production beyond target is usually heavily penalized, but sometimes desired)
    def setupSolverVariables(self, project):
        self.extra_inputs = list(project.inputs - project.explicit_inputs)

        self.desired_byproducts = (project.outputs - project.inputs) - set(project.targets.keys())
        self.minimized_byproducts = set(project.variables) - self.desired_byproducts

        def var_dict_for(names, **kwargs):
            var = cp.Variable(len(names), **kwargs) if len(names) > 0 else []
            # Fun fact, vars is not iterable but you can get items from it by index
            return var, dict(zip(names, [var[i] for i in range(len(names))]))

        self.recipe_vars, self.recipe_vars_dict = var_dict_for(project.recipe_names, name="recipes", nonneg=True)
        self.explicit_in_amounts, self.explicit_in_amounts_dict = var_dict_for(
            project.explicit_inputs, name="expl_in", nonneg=True
        )
        self.extra_in_subs, self.extra_in_sub_dict = var_dict_for(self.extra_inputs, name="in_sub", nonneg=True)
        self.extra_in_switches, self.extra_in_switch_dict = var_dict_for(
            self.extra_inputs, name="in_switch", boolean=True
        )
        self.surplus_amounts, self.surplus_amounts_dict = var_dict_for(project.variables, name="surplus", nonneg=True)

    # Objective function in this case is cost, including recipe tax and prioritized cost variables
    def setupLPProblem(self, project: LpProject, scaled_matrix: LpScaledMatrix):
        # There is some funniness here with the priority ratio.
        # It needs to be large enough that e.g. any possible amount of an explicit input is used 
        # before a minimized byproduct is produced.
        # But it needs to be small because large ratios (which get cubed or more) cause numerical instability.
        # Scaling helps dramatically reduce this number, but it's still good to minimize.
        # The contrived "problem situation" is a scenario looks like this (using unscaled numbers):
        # Biggest-number input (1000 units of A) produces smallest-number output (0.1 B)
        # Second-biggest-number input (999 B) produces second-smallest-number output (0.2 C)
        # ... and so on. Those two would require (999/0.2*1000/0.1) = 49950000.0 A to produce 1 unit of C
        # Solving this problem with no room for unexpected behavior would require a priority ratio of at least that.
        # The "heuristic" I'm using is to square the ratio to allow for at most two such recipe-steps (rounded up)
        # In practical GTNH problems (at least the ones I've seen), this is more than enough.
        priority_ratio = scaled_matrix.max_min_ratio**2
        # Since we're assuming this number is the largest number of any item we will use in this problem,
        # we can also use it as our BIG_NUMBER for the additional input switches (max amount of an input we might use)
        BIG_NUMBER = priority_ratio

        self.checkWarnings(project, scaled_matrix, priority_ratio)

        constraints = []
        for item, item_coeffs, target in zip(project.variables, project.ing_matrix, project.target_vector):
            if item in project.explicit_inputs:
                input_term = self.explicit_in_amounts_dict[item]
            elif item in self.extra_inputs:
                input_term = (self.extra_in_switch_dict[item] - self.extra_in_sub_dict[item]) * BIG_NUMBER
                constraints.append(self.extra_in_switch_dict[item] - self.extra_in_sub_dict[item] >= 0)
            else:
                input_term = 0

            item_eq = (
                cp.sum(cp.multiply(self.recipe_vars, item_coeffs)) + input_term
                == target + self.surplus_amounts_dict[item]
            )
            constraints.append(item_eq)

        desired_byproduct_amounts = [self.surplus_amounts_dict[var] for var in self.desired_byproducts]
        minimized_byproduct_amounts = [self.surplus_amounts_dict[var] for var in self.minimized_byproducts]
        explicit_in_vars = [
            [self.explicit_in_amounts_dict[ing] for ing in group] for group in project.explicit_in_groups
        ]

        explicit_input_term = cp.sum(
            [cp.sum(group) * (priority + 1) for priority, group in enumerate(explicit_in_vars)]
        )

        # I want to make a write-up on this setup, but in short, the list of the solver's priorities is:
        # 1. Maximize the sum of all desired byproducts
        #   - at the lowest priority, this is only done when it does not require inputs
        #   - since we're doing a minimization problem, negating it makes this term "maximized"
        # (2, n-3). Minimize the sum of all explicit inputs, weighted by their priority
        #   - this is the main cost of the problem, and is weighted by the priority of the input
        # n-2. Maximize the sum of all additional input subtractors
        #   - this is half of the additional-input portion of this solver,
        #     which allows for inputs to be added in addition to explicitly provided inputs, at a higher cost.
        #     as such, they are only used after explicit inputs are exhausted.
        #   - As a medium-hacky way to minimize the *number* of additional inputs before the *amount* of each used
        #     while still maintaining a problem with only linear terms (so the formula switch*amount is not allowed),
        #     there is a switch to enable each input and a subtractor to reduce the amount of each input used.
        #     (so the formula is BIG_NUMBER*(switch-sub). To minimize amount used, the sub term is maximized.
        # n-1. Minimize the *number* of additional inputs enabled
        #   - since the subtractors live in the range 0-1, we only need to double the priority multiplier term
        #   - doing so reduces the range of numbers used and saves on our numeric precision budget
        # n. Minimize the sum of all minimized byproducts.
        #   - this is the highest-cost term so that any amount of additional inputs, etc.
        #     are used before these are produced.
        #   - unwanted byproducts should only be produced when strictly necessary to reach a solution.
        #   - otherwise, they should be reprocessed as much as possible.
        #   - once again, switch terms live in the region 0-1,
        #     so we only need to redouble the priority multiplier term.
        objective = (
            priority_ratio**0 * -1 * cp.sum(desired_byproduct_amounts)
            + priority_ratio**1 * explicit_input_term
            + (priority_ratio ** (project.num_priorities + 1) * -1 * cp.sum(self.extra_in_subs))
            + (priority_ratio ** (project.num_priorities + 1) * 2 * cp.sum(self.extra_in_switches))
            + (priority_ratio ** (project.num_priorities + 1) * 4 * cp.sum(minimized_byproduct_amounts))
        )

        return cp.Problem(cp.Minimize(objective), constraints)

    def checkWarnings(self, project: LpProject, scaled_matrix: LpScaledMatrix, priority_ratio):
        def warn(message):
            self.graph.parent_context.log.warn(colored(message, "red"))

        # This warning is based on the ~15 digit precision of doubles, with a bit of a buffer.
        # It uses the formula for the largest priority multiplier in the objective function.
        max_multiplier = priority_ratio ** (1 + project.num_priorities) * 4
        if max_multiplier > 1e11:
            warn(
                f"""Warning: Priority ratio is {priority_ratio},
                which (with your input priority setup) will result in a max priority multiplier of {max_multiplier}. 
                This may cause numerical instability.""",
            )
            if project.num_priorities > 2:
                warn(
                    f"""You seem to have a lot of distinct input priorities ({project.num_priorities}). 
                    Consider grouping some into the same priority level to reduce this number."""
                )
            if scaled_matrix.max_min_ratio > 100:
                nonzero_coeffs = np.abs(project.ing_matrix[project.ing_matrix != 0])
                og_ratio = np.max(nonzero_coeffs) / np.min(nonzero_coeffs)
                warn(
                    f"""The recipe matrix scaler struggled to reduce the Priority Ratio
                    (the ratio between the smallest and largest coefficient in the problem)
                    for your project. It achieved this ratio: {scaled_matrix.max_min_ratio}, from the unscaled ratio: {og_ratio}
                    This will probably cause numerical issues and wierd/incorrect/failed solves.
                    If you actually encountered this issue with a real (not contrived) GTNH problem, please contact the devs.
                    To combat this right now, you will need to find ways to reduce the complexity of the problem
                    (particularly loops and scenarios where the same item is an input/output many times).
                    You can also see if you can find and remove some steps that involve
                    a very large input for a very small output, or vice versa - 
                    however, these scenarios are usually well-handled by the scaler."""
                )


    def run(self):
        # Map from ingredient name to priority level (normalized to 1,2, ...., n) for cost function
        self.project = LpProject.fromGraph(self.graph)
        self.setupSolverVariables(self.project)
        scaled_matrix = LpScaledMatrix(self.project.ing_matrix, self.project.target_vector)
        problem = self.setupLPProblem(self.project, scaled_matrix)
        problem.solve()

def graphPreProcessing(self):
    connectGraph(self)
    # if not self.graph_config.get('KEEP_BACK_EDGES', False):
    #     removeBackEdges(self)
    # Graph.createAdjacencyList(self)


def linearProgrammingSolver(self: ProgramContext, project_name: str, recipes: list[Recipe], graph_config: dict):
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
