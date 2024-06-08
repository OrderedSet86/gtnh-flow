import re
from collections import Counter

from ..graph import Graph
from collections import Counter


class LpProject:
    def __init__(
        self,
        inputs,
        explicit_inputs,
        outputs,
        targets,
        variables,
        recipe_names,
        ing_matrix,
        target_vector,
    ):
        self.inputs = inputs
        self.explicit_inputs = explicit_inputs
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
    def fromRecipes(recipes):
        inputs = set()
        explicit_inputs = set()
        outputs = set()
        targets = {}
        # Just use indices to identify recipes here
        for recipe, io in enumerate(recipes):
            for ing in io.I:
                ing_name = stripBrackets(ing.name)
                inputs.add(ing_name)
            for out in io.O:
                out_name = stripBrackets(out.name)
                outputs.add(out_name)
            if hasattr(io, "cost"):
                for explicit_input in getattr(io, "cost"):
                    explicit_inputs.add(explicit_input)
            if hasattr(io, "target"):
                for target, quant in getattr(io, "target").items():
                    targets[target] = quant

        if len(targets) == 0:
            raise RuntimeError("No targets found! At least one is required for LP solver.")
        if len(explicit_inputs) == 0:
            raise RuntimeError("No explicit inputs found! At least one main ingredient is recommended for reasonable results.")
        
        if not all(target in outputs for target in targets):
            raise RuntimeError(
                "Encountered target which is never an output (likely a spelling mistake). targets: "
                + str(targets)
            )
        if not all(cost in inputs for cost in explicit_inputs):
            raise RuntimeError(
                "Encountered cost/explicit input which is never an input (likely a spelling mistake). costs: "
                + str(explicit_inputs)
            )
    

        variables = list(inputs | outputs)

        recipe_vectors = []
        variable_indices = {var: i for i, var in enumerate(variables)}
        print(variable_indices)
        for recipe in recipes:
            vector = [0] * len(variables)
            for ing in recipe.I:
                vector[variable_indices[stripBrackets(ing.name)]] = -1 * ing.quant
            for out in recipe.O:
                vector[variable_indices[stripBrackets(out.name)]] = out.quant
            recipe_vectors.append(vector)
        recipe_names = LpProject.genRecipeNames(recipes)
        print("Recipe Vectors", list(zip(recipe_names, recipe_vectors)))
        ing_vectors = list(
            zip(*recipe_vectors)
        )  # Transpose (constraints are per-item, not per-recipe)
        target_vector = [targets.get(var, 0) for var in variables]

        return LpProject(
            inputs,
            explicit_inputs,
            outputs,
            targets,
            variables,
            recipe_names,
            ing_vectors,
            target_vector,
        )

    @staticmethod
    def fromGraph(graph: Graph):
        return LpProject.fromRecipes(graph.recipes.values())
