import cvxpy as cp
import numpy as np


class LpScaledMatrix:
    def __init__(self, matrix, target_vector=None):
        self.matrix = matrix
        self.target_vector = target_vector
        has_target = target_vector is not None

        # Augment the matrix with the target vector
        if has_target:
            aug_matrix = np.hstack([matrix, np.array(target_vector).reshape((-1, 1))])
        else:
            aug_matrix = matrix

        num_rows = len(aug_matrix)
        num_cols = len(aug_matrix[0])

        # Variables: scaling factors for each row and column
        item_scales_vars = cp.Variable(num_rows, pos=True)
        recipe_scales_vars = cp.Variable(num_cols, pos=True)

        # Constraints and objective function
        constraints = []
        max_val = cp.Variable(pos=True)
        min_val = cp.Variable(pos=True)

        # The way I set up prioritization requires values >=1, so le's normalize that way
        constraints.append(min_val == 1)

        # Define constraints and objective for each element in the matrix
        for i in range(num_rows):
            for j in range(num_cols):
                if abs(aug_matrix[i][j]) > 0:
                    scaled_value = (
                        abs(aug_matrix[i][j])
                        * item_scales_vars[i]
                        * recipe_scales_vars[j]
                    )
                    # scaled_value = abs(matrix[i][j]) * item_scales[i]
                    constraints.append(max_val >= scaled_value)
                    constraints.append(min_val <= scaled_value)

        objective = cp.Minimize(max_val / min_val)

        # Create the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(gp=True)
        # print("Matrix Normalization Problem in: ", problem.solver_stats.solve_time)

        # Extract the scale factors
        self.item_scales = item_scales_vars.value
        self.recipe_scales = recipe_scales_vars.value[:(-1 if has_target else None)]
        self.target_scale = recipe_scales_vars.value[-1] if has_target else None

        # Apply the scaling factors to the normalized matrix
        scaled_aug_matrix = np.multiply(
            aug_matrix, np.outer(self.item_scales, recipe_scales_vars.value)
        )
        # Un-augment the matrix
        self.scaled_matrix = scaled_aug_matrix[:, :(-1 if has_target else None)]
        self.scaled_target_vector = scaled_aug_matrix[:, -1] if has_target else None
        
        self.max_min_ratio = max_val.value / min_val.value

    def unscale_recipes(self, scaled_recipe_vars):
        target_scale = self.target_scale or 1
        return scaled_recipe_vars * self.recipe_scales / target_scale