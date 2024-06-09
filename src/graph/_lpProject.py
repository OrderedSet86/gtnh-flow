import re
from collections import Counter

from ..graph import Graph
from ._portNodes import stripBrackets
from termcolor import colored
import numpy as np


class LpProject:
    def __init__(
        self,
        inputs: set,
        explicit_input_priorities: dict[str, int],
        outputs: set,
        targets: dict[str, int],
        variables: list[str],
        recipe_names: list[str],
        ing_matrix: np.ndarray,
        target_vector: np.ndarray,
    ):
        self.inputs = inputs
        self.explicit_inputs = explicit_input_priorities.values()
        self.explicit_input_priorities = explicit_input_priorities
        self.num_priorities = len(self.explicit_inputs)
        self.explicit_in_groups = [[]] * self.num_priorities
        for ing, priority in self.explicit_input_priorities.items():
            self.explicit_in_groups[priority].append(ing)
        self.outputs = outputs
        self.targets = targets
        self.variables = variables
        self.recipe_names = recipe_names
        self.ing_matrix = ing_matrix
        self.target_vector = target_vector

    @staticmethod
    def genSlug(s, try_acronym=False):
        slug = re.sub(r"[^a-zA-Z0-9_]", "", re.sub(r"\W+", "_", s.lower()))
        words = slug.split("_")
        if len(slug) > 40 or (try_acronym and len(words) >= 2):
            return "".join(word[0] for word in words if len(word) > 0)
        else:
            return slug[:25]

    @staticmethod
    def genRecipeNames(recipes):
        counter = Counter()
        names = []
        for recipe in recipes:
            represent_ing = recipe.I[0].name if len(recipe.I) > 0 else "nothing"
            name = (
                LpProject.genSlug(recipe.machine, True)
                + "_"
                + LpProject.genSlug(represent_ing)
            )
            counter[name] += 1
            if counter[name] > 1:
                name += f"_{counter[name]}"
            names.append(name)
        return names

    @staticmethod
    def fromGraph(graph: Graph):
        recipes = graph.recipes.values()
        inputs = set()
        # explicit_inputs = set()
        explicit_input_priorities = {}
        outputs = set()
        targets = {}
        # Just use indices to identify recipes here
        for recipe, io in enumerate(recipes):
            for ing in io.I:
                ing_name = stripBrackets(None, ing.name, True)
                inputs.add(ing_name)
            for out in io.O:
                out_name = stripBrackets(None, out.name, True)
                outputs.add(out_name)
            if hasattr(io, "cost"):
                costs = getattr(io, "cost")
                for explicit_input in costs:
                    if explicit_input in explicit_input_priorities:
                        graph.parent_context.log.warn(
                            colored(
                                f"Encountered duplicate explicit input {explicit_input}.",
                                "red",
                            )
                        )
                    explicit_input_priorities[explicit_input] = costs[explicit_input]
            if hasattr(io, "target"):
                for target, quant in getattr(io, "target").items():
                    targets[target] = quant

        if len(targets) == 0:
            raise RuntimeError(
                "No targets found! At least one is required for LP solver."
            )
        if len(explicit_input_priorities.keys()) == 0:
            graph.parent_context.log.warn(
                colored(
                    "No explicit inputs found! At least one main ingredient is highly recommended for reasonable results.",
                    "red",
                )
            )

        if not all(target in outputs for target in targets):
            raise RuntimeError(
                "Encountered target which is never an output (likely a spelling mistake). targets: "
                + str(targets)
            )
        if not all(cost in inputs for cost in set(explicit_input_priorities.keys())):
            raise RuntimeError(
                "Encountered cost/explicit input which is never an input (likely a spelling mistake). costs: "
                + str(list(explicit_input_priorities.keys()))
            )

        variables = list(inputs | outputs)

        recipe_matrix = np.zeros((len(recipes), len(variables)))
        variable_indices = {var: i for i, var in enumerate(variables)}
        print(variable_indices)
        for i, recipe in enumerate(recipes):
            for ing in recipe.I:
                recipe_matrix[
                    i, variable_indices[stripBrackets(None, ing.name, True)]
                ] = (-1 * ing.quant)
            for out in recipe.O:
                recipe_matrix[
                    i, variable_indices[stripBrackets(None, out.name, True)]
                ] = out.quant
        recipe_names = LpProject.genRecipeNames(recipes)
        print("Recipe Matrix", recipe_matrix)
        ing_matrix = (
            recipe_matrix.T
        )  # Transpose (constraints are per-item, not per-recipe)
        target_vector = np.array([targets.get(var, 0) for var in variables])

        return LpProject(
            inputs,
            explicit_input_priorities,
            outputs,
            targets,
            variables,
            recipe_names,
            ing_matrix,
            target_vector,
        )
