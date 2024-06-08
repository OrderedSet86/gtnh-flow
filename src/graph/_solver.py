# In theory solving the machine flow as a linear program is fast and simple -
# this prototype explores this.

import json
import logging
from collections import Counter, deque
from math import isclose

from sympy import linsolve, symbols
from sympy.core.numbers import Float
from sympy.solvers import solve
from sympy.sets.sets import EmptySet
from termcolor import colored

from src.graph import Graph
from src.graph._preProcessing import (
    connectGraph,
    removeBackEdges,
)
from src.graph._postProcessing import (
    addMachineMultipliers,
    addPowerLineNodesV2,
    addSummaryNode,
    addUserNodeColor,
    bottleneckPrint,
    createMachineLabels,
)
from src.graph._output import (
    outputGraphviz
)



class SympySolver:


    def __init__(self, graph):
        self.graph = graph
        self.variables = []
        self.variable_idx_counter = 0 # Autogen current "head" index for variable number
        self.system = []
        self.solved_vars = None # Result from linear solver

        # Lookup is ground truth, efpti is just a convenient way of looking it up
        self.lookup = {} # (machine, product, direction, multi_idx) -> variable index
        self.edge_from_perspective_to_index = {} # (edge, machine_id) -> variable index


    def arrayIndex(self, machine, product, direction, multi_idx=0):
        key = (machine, product, direction, multi_idx)
        if key not in self.lookup:
            self.graph.parent_context.log.debug(colored(f'Addding new variable {key} {self.variable_idx_counter}', 'red'))

            new_variable_name = f'v{self.variable_idx_counter}'
            self.variables.append(symbols(new_variable_name, positive=True, real=True))

            self.lookup[key] = self.variable_idx_counter
            self.variable_idx_counter += 1
            return self.variable_idx_counter - 1
        else:
            return self.lookup[key]


    def run(self):
        # Construct system of equations
        self._addUserLocking() # add known equations from user "number" and "target" args
        self._addMachineInternalLocking() # add relations inside machines - eg 1000 wood tar -> 350 benzene
        self._populateEFPTI() # construct "edge_from_perspective_to_index" - a useful index lookup for next steps
        if self.graph.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
            self._debugAddVarsToEdges()
            self.outputGraphvizProxy()

        self.graph.parent_context.log.debug(colored(f'Adding machine-machine edges', 'yellow'))
        self._addMachineMachineEdges() # add equations between machines, including complex situations - eg multi IO

        # Solve and if unsolvable, adjust until it is
        self._solve()

        if self.solved_vars:
            self._writeQuantsToGraph()
    
    def outputGraphvizProxy(self):
        outputGraphviz(self.graph)
        if self.graph.graph_config.get('PRINT_BOTTLENECKS') and self.solved_vars:
            bottleneckPrint(self.graph)

    def _addNewEquation(self, new_eqn):
        self.graph.parent_context.log.debug(colored(f'New equation: {new_eqn}', 'cyan'))
        self.system.append(new_eqn)


    def _addUserLocking(self):
        # Add user-determined locked inputs
        targeted_nodes = [i for i, x in self.graph.recipes.items() if getattr(x, 'target', False) != False]
        numbered_nodes = [i for i, x in self.graph.recipes.items() if getattr(x, 'number', False) != False]

        ln = len(numbered_nodes)
        lt = len(targeted_nodes)

        if lt == 0 and ln == 0:
            raise RuntimeError('Need at least one "number" or "target" argument to base machine balancing around.')

        elif ln != 0 or lt != 0:
            # Add numbered nodes
            for rec_id in numbered_nodes:
                rec = self.graph.recipes[rec_id]

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
                solved_quant_s = core_ing.quant * rec.number / (rec.dur/20)
                new_eqn = (
                    self.variables[self.arrayIndex(rec_id, core_ing.name, core_direction)]
                    -
                    solved_quant_s
                )
                self._addNewEquation(new_eqn)

            # Add targetted nodes
            for rec_id in targeted_nodes:
                rec = self.graph.recipes[rec_id]
                if len(rec.target) > 1:
                    raise NotImplementedError('Currently only one targeted ingredient per machine - feel free to open Github ticket')
                target_ingredient = list(rec.target)[0]
                target_quant = rec.target[target_ingredient]

                # Look up the exact ingredient and add the constant to the system of equations
                for ing_direction in ['I', 'O']:
                    directional_matches = [x.name for x in getattr(rec, ing_direction)._ings if x.name == target_ingredient]

                    if directional_matches:
                        ing_name = directional_matches[0]
                        new_eqn = (
                            self.variables[self.arrayIndex(rec_id, ing_name, ing_direction)]
                            -
                            target_quant
                        )
                        self._addNewEquation(new_eqn)
                        break
                else:
                    raise RuntimeError(f'Targetted quantity must be in machine I/O for \n{rec}')

    
    def _addMachineInternalLocking(self):
        # Add machine equations
        for rec_id in self.graph.nodes:
            if self.graph._checkIfMachine(rec_id):
                rec = self.graph.recipes[rec_id]
                
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
                            new_eqn = (
                                self.variables[self.arrayIndex(rec_id, core_ing.name, core_direction)]
                                -
                                multiple * self.variables[self.arrayIndex(rec_id, ing.name, ing_direction)]
                            )
                            self._addNewEquation(new_eqn)


    def _populateEFPTI(self):
        # Populate edge_from_perspective_to_index for all edges - so there's something consistent to call for all edges
        for edge in self.graph.edges:
            a, b, product = edge

            if self.graph._checkIfMachine(a):
                if (edge, a) not in self.edge_from_perspective_to_index:
                    self.edge_from_perspective_to_index[(edge, a)] = self.arrayIndex(a, product, 'O')

            if self.graph._checkIfMachine(b):
                if (edge, b) not in self.edge_from_perspective_to_index:
                    self.edge_from_perspective_to_index[(edge, b)] = self.arrayIndex(b, product, 'I')


    def _addMachineMachineEdges(self):
        # Add machine-machine edges
        # Need to be careful about how these are added - multi input and multi output can
        #   require arbitrarily many variables per equation
        # See https://github.com/OrderedSet86/gtnh-flow/issues/7#issuecomment-1331312996 for an example
        # Solution is below the linked comment
        computed_edges = set()
        for edge in self.graph.edges:
            if edge in computed_edges:
                continue
            a, b, product = edge
            if self.graph._checkIfMachine(a) and self.graph._checkIfMachine(b):
                # print(f'Machine edge detected! {edge}')

                # Run DFS to find all connected machine-machine edges using the same product
                involved_machines = Counter()
                involved_machine_edges = set()
                q = [edge]
                while q:
                    dfs_edge = q.pop()
                    if dfs_edge in involved_machine_edges:
                        continue
                    dfs_a, dfs_b, _ = dfs_edge

                    involved_machine_edges.add(dfs_edge)
                    involved_machines[dfs_a] += 1
                    involved_machines[dfs_b] += 1

                    # Check for all adjacent I/O edges using the same product
                    for out_edge in self.graph.adj_machine[a]['O']: # Multi-output
                        if out_edge[2] == product:
                            q.append(out_edge)
                    for in_edge in self.graph.adj_machine[b]['I']: # Multi-input
                        if in_edge[2] == product:
                            q.append(in_edge)
                
                if len(involved_machine_edges) == 1:
                    # Simple version - all A output fulfills all B input

                    self._addNewEquation(
                        self.variables[self.arrayIndex(a, product, 'O')]
                        -
                        self.variables[self.arrayIndex(b, product, 'I')]
                    )
                else:
                    # Hard version - A and B fulfill some percentage of each other and other machines in a network
                    # Each multi-input and multi-output will require the creation of minimum 2 new variables
                    # print(involved_machines)
                    # print(involved_edges)
                    involved_machine_edges = list(involved_machine_edges)

                    complex_machine_id = involved_machines.most_common()[0][0]
                    multi_io_direction = 'O' if involved_machine_edges[0][0] == complex_machine_id else 'I'
                    self._addMultiEquationsOnEdge(complex_machine_id, multi_io_direction, involved_machine_edges)

                computed_edges.update(set(involved_machine_edges))


    def _addMultiEquationsOnEdge(self, multi_machine, multi_io_direction, involved_machine_edges):
        # Log
        multi_product = involved_machine_edges[0][2]
        if multi_io_direction == 'O':
            self.graph.parent_context.log.info(colored(f'Solving multi-output scenario involving {multi_product}!', 'green'))
        elif multi_io_direction == 'I':
            self.graph.parent_context.log.info(colored(f'Solving multi-input scenario involving {multi_product}!', 'green'))

        self.graph.parent_context.log.debug(
            colored(
                f'Old base var: {self.arrayIndex(multi_machine, multi_product, multi_io_direction, multi_idx=0)}',
                'cyan'
            )
        )

        multi_idx = 1 # starts at 1 for new variables
        new_variables = []
        for edge in involved_machine_edges:
            other_machine_id = edge[0] if edge[1] == multi_machine else edge[1]

            # Add new variables and update efpti
            new_var_index = self.arrayIndex(multi_machine, multi_product, multi_io_direction, multi_idx=multi_idx)
            new_variables.append(self.variables[new_var_index])
            multi_idx += 1
            self.edge_from_perspective_to_index[(edge, multi_machine)] = new_var_index

            # Add "simple" machine-machine equations
            other_var_index = self.edge_from_perspective_to_index[(edge, other_machine_id)]
            if multi_io_direction == 'O':
                self._addNewEquation(
                    self.variables[new_var_index]
                    -
                    self.variables[other_var_index]
                )
            elif multi_io_direction == 'I':
                self._addNewEquation(
                    self.variables[other_var_index]
                    -
                    self.variables[new_var_index]
                )

        # Add new overall balancing equation for multi-IO
        eqn = self.variables[self.arrayIndex(multi_machine, multi_product, multi_io_direction, multi_idx=0)]
        for newvar in new_variables:
            eqn -= newvar
        self._addNewEquation(eqn)


    def _solve(self):
        # while True: # Loop until solved - algorithm may adjust edges each time it sees an EmptySet
        res = linsolve(self.system, self.variables)
        if isinstance(res, EmptySet):
            self.graph.parent_context.log.warning(colored('EmptySet response - likely overdetermined', 'red'))

            # Check for water in inputs
            for edge in self.graph.edges:
                a, b, product = edge
                if product == 'water':
                    if self.graph._checkIfMachine(a) and self.graph._checkIfMachine(b):
                        self.graph.parent_context.log.warning(colored('Water detected on machine/machine edge - possible cause of overdetermination', 'yellow'))
                        break
            self._searchForInconsistency()
            return

        lstres = list(res)
        if len(lstres) > 1:
            raise NotImplementedError('Multiple solutions - no code written to deal with this scenario yet')
        self.solved_vars = res.args[0]
        any_unsolved = False
        for i, var in enumerate(self.solved_vars):
            if not isinstance(var, Float):
                any_unsolved = True
                print(i, var)
        if any_unsolved:
            print('all soln:', res)

            self._debugAddVarsToEdges()
            self.outputGraphvizProxy()

            raise RuntimeError(
                '\n    Unsolved variables - machine system is underdefined.'
                '\n    Likely cause is either disconnected machines or too little information.'
            )


    def _searchForInconsistency(self):
        # Solve each equation stepwise until inconsistency is found, then report to end user

        self.graph.parent_context.log.info(colored('Searching for inconsistency in system of equations...', 'blue'))

        equations_to_check = deque(self.system)
        max_iter = len(self.system) ** 2 + 1
        iterations = 0

        solved_values = {}
        inconsistent_variables = []

        while equations_to_check and iterations < max_iter:
            expr = equations_to_check.popleft()

            # Detect variable or variables in equation
            involved_variables = expr.as_terms()[-1]

            # Solve if feasible, otherwise go to next
            # Can be solved if only 1 unknown variable
            unsolved = [x for x in involved_variables if x not in solved_values]
            solved = [x for x in involved_variables if x in solved_values]

            if len(unsolved) == 1:
                for var in solved:
                    expr = expr.subs(var, solved_values[var])
                # print('   ', expr)
                
                # If expr is a nonzero constant, inconsistency is found
                if expr.is_constant():
                    constant_diff = float(expr)
                    if not isclose(constant_diff, 0.0, abs_tol=0.00000001):
                        self.graph.parent_context.log.warning(colored(f'Inconsistency found with {involved_variables}! (diff = {constant_diff})', 'red'))
                        # Now print what the variables are referring to
                        involved_varindex = [int(str(x)[1:]) for x in involved_variables]
                        for edge_perspective_data, varindex in self.edge_from_perspective_to_index.items():
                            if varindex in involved_varindex:
                                self.graph.parent_context.log.warning(colored(f'    v{varindex}: {edge_perspective_data}', 'red'))

                        inconsistent_variables.append((involved_variables, constant_diff))
                        iterations += 1
                        continue

                unvar = unsolved[0]
                solution = solve(expr, unvar)
                if len(solution) == 1:
                    sval = solution[0]
                    solved_values[unvar] = sval
                    # print(sval)
                else:
                    # raise NotImplementedError(f'{solution=}')
                    pass # TODO: Handle 0 case

            else:
                equations_to_check.append(expr)
            
            iterations += 1

        # Output graph for end user to view
        self.graph.parent_context.log.warning(colored(f'Refer to graph for more information.', 'red'))
        self._debugAddVarsToEdges()
        self.outputGraphvizProxy()
        
        if inconsistent_variables:
            raise NotImplementedError('Both linear and nonlinear solver found empty set, so system of equations has no solutions -- report to dev.')

        if len(unsolved) == 0:
            return

        # Check inconsistent equations to see if products on both sides are the same - these are the core issues
        def var_to_idx(var):
            return int(str(var).strip('v'))

        # for k, v in edge_from_perspective_to_index.items():
        #     print(k, v)

        idx_to_mpdm = {idx: mpdm for mpdm, idx in self.lookup.items()}
        for group, constant_diff in inconsistent_variables:
            assert len(group) == 2 # TODO: NotImplemented

            # Reverse lookup using var
            products = set()
            mpdm_cache = []
            for var in group:
                idx = var_to_idx(var)
                mpdm = idx_to_mpdm[idx]
                mpdm_cache.append(mpdm)
                machine, product, direction, multi_idx = mpdm
                products.add(product)
            
            # When problematic inconsistency is found...
            if len(products) == 1:
                self.graph.parent_context.log.warning(colored(f'Major inconsistency: {group}', 'red'))

                self.graph.parent_context.log.warning(colored(f'Between output={self.graph.recipes[mpdm_cache[0][0]].O}', 'red'))
                self.graph.parent_context.log.warning(colored(f'    and  input={self.graph.recipes[mpdm_cache[1][0]].I}', 'red'))

                self.graph.parent_context.log.info(colored('[BETA] Please fix by either:', 'green'))

                # if constant_diff < 0:
                #     parent_group_idx = 0
                #     child_group_idx = 1
                # else:
                parent_group_idx = 0
                child_group_idx = 1

                # Negative means too much of right side, or too few of other sided inputs
                self.graph.parent_context.log.info(colored(f'1. Sending excess "{group[parent_group_idx]} {product}" to sink', 'blue'))

                # Check other sided inputs
                machine, product, direction, multi_idx = idx_to_mpdm[var_to_idx(group[child_group_idx])]
                self.graph.parent_context.log.info(colored(f'2. Pulling more "{group[child_group_idx]} {product}" from source', 'blue'))

                # nonself_product = []
                # for edge in self.graph.adj[machine][direction]:
                #     # print(self.graph.adj[machine])
                #     # print(edge)
                #     a, b, edgeproduct = edge
                #     if edgeproduct != product:
                #         nonself_product.append((
                #             edgeproduct,
                #             'v' + f'{self.edge_from_perspective_to_index[(edge, machine)]}',
                #         ))

                # self.graph.parent_context.log.info(colored(f'2. Pulling more {nonself_product} from source', 'blue'))

                # # Output graph for end user to view
                # self._debugAddVarsToEdges()
                # self.outputGraphvizProxy()

                # TODO: Automate solution process fully

                # selection = input() # TODO: Verify input
                selection = ''

                if selection == '1':
                    # Send excess to sink
                    # 1. Similar to multi-IO: (a-c could probably be spun off into another fxn)
                    #       a. reassociate old variable with machine sum of product
                    #       b. create a new variable for old edge
                    #       c. create a new variable for machine -> sink
                    # 2. Redo linear solve
                    # 3. Give option for user to add new I/O association to YAML config (will delete comments)
                    pass
                elif selection == '2':
                    # Pull more of each other input from source
                    # 1. Similar to multi-IO: (a-c could probably be spun off into another fxn)
                    #       a. reassociate each old variable on all sides of machine with machine sum of product
                    #       b. create a new variable for each old edge
                    #       c. create a new variable for each source -> machine
                    # 2. Redo linear solve
                    # 3. Give option for user to add new I/O association to YAML config (will delete comments)
                    pass


    def _debugAddVarsToEdges(self):
        # Add variable indices to edges and rec_id to machines

        # Lookup is a dictionary defined like this:
        #   (machine, product, direction, multi_idx) -> variable index
        # Edges in self.edges are defined like:
        #   rec_id_a, rec_id_b, product
        
        # Add edge debug variables
        for node_id in self.graph.nodes:
            if self.graph._checkIfMachine(node_id):
                rec_id = node_id
                rec = self.graph.recipes[rec_id]
            else:
                continue

            for direction in ['I', 'O']:
                adjacent_edges = self.graph.adj[rec_id][direction]
                adjacent_machine_edges = self.graph.adj_machine[rec_id][direction]
                for ing in getattr(rec, direction):
                    relevant_edges = [edge for edge in adjacent_edges if edge[2] == ing.name]
                    relevant_machine_edges = [edge for edge in adjacent_machine_edges if edge[2] == ing.name]

                    if len(relevant_edges) == 0:
                        raise RuntimeError(f'No edge found for ingredient {ing.name} in {rec_id} {direction}!')
                    elif len(relevant_machine_edges) > 1:
                        # print(f'Multi-IO machine detected! {rec_id} {direction} {ing.name} {relevant_edges}')
                        
                        # Make assumption that any equation involving all adjacent edge variables is the multi-IO base variable
                        adjacent_variables = []
                        for edge in relevant_machine_edges:
                            perspective = rec_id
                            edge_perspective_data = (edge, perspective)
                            variableIndex = self.edge_from_perspective_to_index[edge_perspective_data]
                            adjacent_variables.append(variableIndex)

                        own_ingredient = relevant_machine_edges[0][2]

                        # print(relevant_edges)
                        # print(adjacent_variables, ing.name)

                        # Get related equation and base variable
                        for i, eq in enumerate(self.system):
                            eqn_var_indices = [int(str(x)[1:]) for x in eq.as_terms()[-1]]
                            if all([x in eqn_var_indices for x in adjacent_variables]):
                                # print(f'!! Associated equation: {eq}')
                                assoc_eqn = (eq, eqn_var_indices)
                                break
                        base_var_idx = set(assoc_eqn[1]) - set(adjacent_variables)
                        base_var_idx = list(base_var_idx)[0]

                        # Add base_var_idx to ingredient name "own_ingredient" in node "rec_id"
                        # TODO:

                        # Get all var indices
                        var_indices = []
                        for edge in relevant_edges:
                            perspective = rec_id
                            edge_perspective_data = (edge, perspective)
                            variableIndex = self.edge_from_perspective_to_index[edge_perspective_data]
                            var_indices.append(variableIndex)
                        var_string = f"[{', '.join([f'v{x}' for x in var_indices])}]"

                        # Add ALL variable indices ONCE to ONE edge
                        info_edge = relevant_edges[0]
                        perspective = rec_id
                        variableIndex = self.edge_from_perspective_to_index[(info_edge, perspective)]
                        a, b, product = info_edge

                        if 'debugHead' not in self.graph.edges[info_edge]:
                            self.graph.edges[info_edge]['debugHead'] = ''
                        if 'debugTail' not in self.graph.edges[info_edge]:
                            self.graph.edges[info_edge]['debugTail'] = ''

                        if perspective == b:
                            self.graph.edges[info_edge]['debugHead'] += f'{var_string}'
                        elif perspective == a:
                            self.graph.edges[info_edge]['debugTail'] += f'{var_string}'

                    elif len(relevant_edges) >= 1:
                        edge = relevant_edges[0]
                        perspective = rec_id
                        edge_perspective_data = (edge, perspective)
                        variableIndex = self.edge_from_perspective_to_index[edge_perspective_data]
                        a, b, product = edge
                        
                        if 'debugHead' not in self.graph.edges[edge]:
                            self.graph.edges[edge]['debugHead'] = ''
                        if 'debugTail' not in self.graph.edges[edge]:
                            self.graph.edges[edge]['debugTail'] = ''

                        if perspective == b:
                            self.graph.edges[edge]['debugHead'] += f'v{variableIndex}'
                        elif perspective == a:
                            self.graph.edges[edge]['debugTail'] += f'v{variableIndex}'

        # Add machine debug variables
        for node_id in self.graph.nodes:
            if self.graph._checkIfMachine(node_id):
                rec_id = node_id
                rec = self.graph.recipes[rec_id]
            else:
                continue

            self.graph.nodes[rec_id]['label'] = f'[id:{rec_id}] {rec.machine}'


    def _writeQuantsToGraph(self):
        # Update graph edge values
        for edge in self.graph.edges:
            a, b, product = edge
            a_machine = self.graph._checkIfMachine(a)
            b_machine = self.graph._checkIfMachine(b)

            if a_machine and b_machine:
                # Sanity check both edges and make sure they match
                a_index = self.edge_from_perspective_to_index[(edge, a)]
                b_index = self.edge_from_perspective_to_index[(edge, b)]
                
                a_quant = self.solved_vars[a_index]
                b_quant = self.solved_vars[b_index]

                if isclose(a_quant, b_quant, rel_tol=0.05):
                    relevant_edge = self.graph.edges[edge]
                    relevant_edge['quant'] = float(a_quant)
                    relevant_edge['locked'] = True # TODO: Legacy - check if can be removed
                else:
                    raise RuntimeError('\n'.join([
                        'Mismatched machine-edge quantities:',
                        f'{a_quant}',
                        f'{b_quant}',
                    ]))

            elif a_machine or b_machine:
                # Assume a_machine for now
                if a_machine:
                    solution_index = self.edge_from_perspective_to_index[(edge, a)]
                elif b_machine:
                    solution_index = self.edge_from_perspective_to_index[(edge, b)]

                quant = self.solved_vars[solution_index]
                relevant_edge = self.graph.edges[edge]
                relevant_edge['quant'] = float(quant)
                relevant_edge['locked'] = True # TODO: Legacy - check if can be removed



def graphPreProcessing(self):
    connectGraph(self)
    if not self.graph_config.get('KEEP_BACK_EDGES', False):
        removeBackEdges(self)
    self.createAdjacencyList()


def graphPostProcessing(self):
    if self.graph_config.get('POWER_LINE', False):
        addPowerLineNodesV2(self)

    addMachineMultipliers(self)
    createMachineLabels(self)
    addSummaryNode(self)
    addUserNodeColor(self)

    if self.graph_config.get('COMBINE_INPUTS', False):
        self._combineInputs()
    if self.graph_config.get('COMBINE_OUTPUTS', False):
        self._combineOutputs()


def systemOfEquationsSolverGraphGen(self, project_name, recipes, graph_config):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    self._graph = g # For test access

    graphPreProcessing(g)

    g.parent_context.log.info(colored('Running linear solver...', 'green'))
    solver = SympySolver(g)
    solver.run()

    if solver.solved_vars:
        graphPostProcessing(g)
        solver.outputGraphvizProxy()


if __name__ == '__main__':
    from factory_graph import ProgramContext
    c = ProgramContext()

    c.run(graph_gen=systemOfEquationsSolverGraphGen)