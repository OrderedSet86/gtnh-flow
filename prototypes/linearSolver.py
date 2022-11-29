# In theory solving the machine flow as a linear program is fast and simple -
# this prototype explores this.

import numpy as np
from sympy import linsolve, symbols

from src.graph import Graph
from factory_graph import ProgramContext


def linearSolver(self):
    # "self" here refers to the graph instance - will later become a class member

    # System of equations:
    # input_per_s_1 = machine_3_count * output_per_s_3 + machine_5_count * output_per_s_5
    # output_per_s_1 = (constant) * input_per_s_1
    # ...

    # Create method of inputting unique products into the right index
    lookup = {}
    idx_counter = 0
    def arrayIndex(machine, product, direction):
        nonlocal lookup, idx_counter
        key = (machine, product, direction)
        if key not in lookup:
            lookup[key] = idx_counter
            idx_counter += 1
            return idx_counter - 1
        else:
            return lookup[key]
    
    # Determine how many unique products there are (for knowing matrix column count)
    num_edges = len(self.edges)

    # Create array for entire chart
    for edge in self.edges:
        row = [0]*num_edges

        # Determine machine edges
        a, b, product = edge
        if self._checkIfMachine(a):
            arrayIndex(a, product, 'O')
            

def linearSympy(self):
    # Compute number of variables
    num_variables = 0
    for rec_id in self.nodes:
        if self._checkIfMachine(rec_id):
            rec = self.recipes[rec_id]
            # one for each input/output ingredient
            num_variables += len(rec.I) + len(rec.O)

    symbols_str = ', '.join([str(x) for x in range(num_variables)])
    variables = symbols(symbols_str)

    # Create rule for which ingredient => which symbol index
    # Need to invert this later on when populating graph
    lookup = {}
    idx_counter = 0
    def arrayIndex(machine, product, direction):
        nonlocal lookup, idx_counter
        key = (machine, product, direction)
        if key not in lookup:
            lookup[key] = idx_counter
            idx_counter += 1
            return idx_counter - 1
        else:
            return lookup[key]

    system = []

    
    # Add user-determined locked inputs
    targeted_nodes = [i for i, x in self.recipes.items() if getattr(x, 'target', False) != False]
    numbered_nodes = [i for i, x in self.recipes.items() if getattr(x, 'number', False) != False]

    ln = len(numbered_nodes)
    lt = len(targeted_nodes)

    if lt == 0 and ln == 0:
        raise RuntimeError('Need at least one "number" or "target" argument to base machine balancing around.')

    elif ln != 0 or lt != 0:
        # Add numbered nodes
        for rec_id in numbered_nodes:
            rec = self.recipes[rec_id]

            # Pick a random ingredient to be the "solved" one, then solve for it based on machine number
            if len(rec.I):
                core_ing = rec.I[0]
                core_direction = 'I'
            elif len(rec.O):
                core_ing = rec.O[0]
                core_direction = 'O'
            else:
                raise RuntimeError(f'{rec} has no inputs or outputs!')

            # Solve for ingredient quantity and add to system of equations
            solved_quant = core_ing.quant * rec.number
            system.append(
                variables[arrayIndex(rec_id, core_ing.name, core_direction)]
                -
                solved_quant
            )

        # Add targetted nodes
        for rec_id in targeted_nodes:
            rec = self.recipes[rec_id]
            if len(rec.target) > 1:
                raise NotImplementedError('Currently only one targeted ingredient per machine - feel free to open Github ticket')
            target_ingredient = list(rec.target)[0]
            target_quant = rec.target[target_ingredient]

            # Look up the exact ingredient and add the constant to the system of equations
            for ing_direction in ['I', 'O']:
                directional_matches = [x.name for x in getattr(rec, ing_direction)._ings if x.name == target_ingredient]

                if directional_matches:
                    ing_name = directional_matches[0]
                    system.append(
                        variables[arrayIndex(rec_id, ing_name, ing_direction)]
                        -
                        target_quant
                    )
                    break
            else:
                raise RuntimeError(f'Targetted quantity must be in machine I/O for \n{rec}')


    # Add machine equations
    for rec_id in self.nodes:
        if self._checkIfMachine(rec_id):
            rec = self.recipes[rec_id]
            
            # Pick a single ingredient to represent the rest of the equations (to avoid having more equations than needed)
            if len(rec.I):
                core_ing = rec.I[0]
                core_direction = 'I'
            elif len(rec.O):
                core_ing = rec.O[0]
                core_direction = 'O'
            else:
                raise RuntimeError(f'{rec} has no inputs or outputs!')

            # Add equations in form of core_ingredient
            for ing_direction in ['I', 'O']:
                for ing in getattr(rec, ing_direction):
                    if ing.name != core_ing.name:
                        # Determine constant multiple between products
                        multiple = core_ing.quant / ing.quant
                        system.append(
                            variables[arrayIndex(rec_id, core_ing.name, core_direction)]
                            -
                            multiple * variables[arrayIndex(rec_id, ing.name, ing_direction)]
                        )


    # Add machine-machine edges
    for edge in self.edges:
        a, b, product = edge
        if self._checkIfMachine(a) and self._checkIfMachine(b):
            print(f'Machine edge detected! {edge}')
            system.append(
                variables[arrayIndex(a, product, 'O')]
                -
                variables[arrayIndex(b, product, 'I')]
            )

    # Do linear solve
    res = linsolve(system, variables)
    lstres = list(res)
    if len(lstres) > 1:
        raise NotImplementedError('Multiple solutions - no code written to deal with this scenario yet')
    solved_vars = res.args[0]

    print(res)

    # Invert lookup to figure out which answers correspond to which ingredients
    # Then update graph accordingly
    for machine_info, index in lookup.items():
        print(machine_info, solved_vars[index])

    for edge in self.edges:
        a, b, product = edge
        a_machine = self._checkIfMachine(a)
        b_machine = self._checkIfMachine(b)

        if a_machine and b_machine:
            # Sanity check both edges and make sure they match
            a_index = lookup[(a, product, 'O')]
            b_index = lookup[(b, product, 'I')]
            
            a_quant = solved_vars[a_index]
            b_quant = solved_vars[b_index]

            if a_quant == b_quant: # TODO: Be less strict about equality
                relevant_edge = self.edges[edge]
                relevant_edge['quant'] = float(a_quant)
                relevant_edge['locked'] = True # TODO: Legacy - check if can be removed
            else:
                # TODO: Increase error verbosity
                raise RuntimeError('Mismatched edges')

        elif a_machine or b_machine:
            # Assume a_machine for now
            if a_machine:
                solution_index = lookup[(a, product, 'O')]
            elif b_machine:
                solution_index = lookup[(b, product, 'I')]

            quant = solved_vars[solution_index]
            relevant_edge = self.edges[edge]
            relevant_edge['quant'] = float(quant)
            relevant_edge['locked'] = True # TODO: Legacy - check if can be removed


def graphPostProcessing(self):
    if self.graph_config.get('POWER_LINE', False):
        self._addPowerLineNodes()
    self._addSummaryNode()

    if self.graph_config.get('COMBINE_INPUTS', False):
        self._combineInputs()
    if self.graph_config.get('COMBINE_OUTPUTS', False):
        self._combineOutputs()


def linearSolverGraphGen(self, project_name, recipes, graph_config):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    g.connectGraph()

    g.removeBackEdges()
    g.createAdjacencyList()

    linearSympy(g)
    graphPostProcessing(g)

    g.outputGraphviz()


if __name__ == '__main__':
    c = ProgramContext()

    c.run(graph_gen=linearSolverGraphGen)