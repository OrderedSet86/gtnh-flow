# In theory solving the machine flow as a linear program is fast and simple -
# this prototype explores this.

from scipy.optimize import linprog

from src.graph import Graph
from factory_graph import ProgramContext


def linearSolver(self):
    # "self" here refers to the graph instance - will later become a class member

    # System of equations:
    # input_per_s_1 = machine_3_count * output_per_s_3 + machine_5_count * output_per_s_5
    # output_per_s_1 = (constant) * input_per_s_1
    # ...

    # output_per_s_1 = (constant) * (machine_3_count * output_per_s3 + machine_5_count * output_per_s3)

    pass




def linearSolverGraphGen(self, project_name, recipes, graph_config):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    g.connectGraph()

    # TODO: Linear solver here
    # g.linearSolver = linearSolver
    # g.linearSolver()

    g.outputGraphviz()


if __name__ == '__main__':
    c = ProgramContext()

    c.run(graph_gen=linearSolverGraphGen)