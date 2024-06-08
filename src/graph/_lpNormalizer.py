import cvxpy as cp
import numpy as np


class LpNormalizedMatrix:
    def __init__(self, matrix, target_vector):
        self.matrix = matrix
        self.target_vector = target_vector

        # Augment the matrix with the target vector
        aug_matrix = np.hstack([matrix, np.array(target_vector).reshape((-1, 1))])

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
        print("Matrix Normalization Problem in: ", problem.solver_stats.solve_time)

        # Extract the scale factors
        self.item_scales = item_scales_vars.value
        self.recipe_scales = recipe_scales_vars.value[:-1]
        self.target_scale = recipe_scales_vars.value[-1]

        # Apply the scaling factors to the normalized matrix
        scaled_aug_matrix = np.multiply(
            aug_matrix, np.outer(self.item_scales, recipe_scales_vars.value)
        )
        # Un-augment the matrix
        self.scaled_matrix = scaled_aug_matrix[:, :-1]
        self.scaled_target_vector = scaled_aug_matrix[:, -1]
