# In theory solving the machine flow as a linear program is fast and simple -
# this prototype explores this.

import logging
from collections import Counter, deque
from copy import deepcopy
from math import isclose
from string import ascii_uppercase

import yaml
from sympy import linsolve, symbols
from sympy.solvers import solve
from sympy.sets.sets import EmptySet

from src.data.basicTypes import Ingredient, IngredientCollection, Recipe
from src.graph import Graph
from src.graph._utils import swapIO



class SympySolver:


    def __init__(self, graph):
        self.graph = graph
        self.variables = []
        self.variable_idx_counter = 0 # Autogen current "head" index for variable number
        self.num_variables = 0 # Expected number of variables - if this diverges from vic, something broke
        self.system = []
        self.solved_vars = None # Result from linear solver

        # TODO: Look into merging lookup and edge_from_perspective_to_index - they describe mostly the same thing
        self.lookup = {} # (machine, product, direction, multi_idx) -> variable index
        self.edge_from_perspective_to_index = {} # (edge, machine_id) -> variable index


    def arrayIndex(self, machine, product, direction, multi_idx=0):
        key = (machine, product, direction, multi_idx)
        if key not in self.lookup:
            # print(f'Unique key {key}')
            self.lookup[key] = self.variable_idx_counter
            self.variable_idx_counter += 1
            return self.variable_idx_counter - 1
        else:
            return self.lookup[key]


    def run(self):
        self._createVariables() # initialize v1, v2, v3 ... (first pass)

        # Construct system of equations
        self._addUserLocking() # add known equations from user "number" and "target" args
        self._addMachineInternalLocking() # add relations inside machines - eg 1000 wood tar -> 350 benzene
        self._populateEFPTI() # construct "edge_from_perspective_to_index" - a useful index lookup for next steps
        self._addMachineMachineEdges() # add equations between machines, including complex situations - eg multi IO

        # Solve and if unsolvable, adjust until it is
        self._solve()

        self._writeQuantsToGraph()


    def _createVariables(self):
        # Compute number of variables
        num_variables = 0
        for rec_id in self.graph.nodes:
            if self.graph._checkIfMachine(rec_id):
                rec = self.graph.recipes[rec_id]
                # one for each input/output ingredient
                num_variables += len(rec.I) + len(rec.O)
        self.num_variables = num_variables

        symbols_str = ', '.join(['v' + str(x) for x in range(num_variables)])
        self.variables = list(symbols(symbols_str, positive=True, real=True))
    
    
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
                self.system.append(
                    self.variables[self.arrayIndex(rec_id, core_ing.name, core_direction)]
                    -
                    solved_quant_s
                )

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
                        self.system.append(
                            self.variables[self.arrayIndex(rec_id, ing_name, ing_direction)]
                            -
                            target_quant
                        )
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
                            self.system.append(
                                self.variables[self.arrayIndex(rec_id, core_ing.name, core_direction)]
                                -
                                multiple * self.variables[self.arrayIndex(rec_id, ing.name, ing_direction)]
                            )


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
                involved_edges = set()
                q = [edge]
                while q:
                    dfs_edge = q.pop()
                    if dfs_edge in involved_edges:
                        continue
                    dfs_a, dfs_b, _ = dfs_edge

                    involved_edges.add(dfs_edge)
                    involved_machines[dfs_a] += 1
                    involved_machines[dfs_b] += 1

                    # Check for all adjacent I/O edges using the same product
                    for edge in self.graph.adj[a]['O']: # Multi-output
                        if edge[2] == product:
                            q.append(edge)
                    for edge in self.graph.adj[b]['I']: # Multi-input
                        if edge[2] == product:
                            q.append(edge)
                
                if len(involved_edges) == 1:
                    # Simple version - all A output fulfills all B input

                    self.system.append(
                        self.variables[self.arrayIndex(a, product, 'O')]
                        -
                        self.variables[self.arrayIndex(b, product, 'I')]
                    )
                else:
                    # Hard version - A and B fulfill some percentage of each other and other machines in a network
                    # Each multi-input and multi-output will require the creation of minimum 2 new variables
                    # print(involved_machines)
                    # print(involved_edges)

                    # Assume no loops since DAG was enforced earlier
                    for rec_id, count in involved_machines.most_common(): # most_common so multi-IO variables are created first
                        if count > 1:
                            # Multi-input or multi-output
                            # Old variable is now the collected amount for that side
                            # Then, add new variable for each input/output (minimum 2)

                            relevant_edge = None
                            destinations = []
                            for edge in involved_edges:
                                multi_a, multi_b, multi_product = edge
                                if rec_id in edge[:2]:
                                    if relevant_edge is None:
                                        relevant_edge = edge
                                    if multi_a == rec_id:
                                        destinations.append(multi_b)
                                    elif multi_b == rec_id:
                                        destinations.append(multi_a)

                            self._addMultiEquationsOnEdge(relevant_edge, rec_id, destinations)

                        else:
                            # Still "simple" - can keep old self.graph variable, but opposite end of edge must point at correct multi-IO variable

                            # print('Pre-efpti')
                            # for k, v in edge_from_perspective_to_index.items():
                            #     print(k, v)
                            # print()

                            # Figure out if simple machine is a or b
                            for edge in involved_edges:
                                if rec_id in edge[:2]:
                                    relevant_edge = edge
                            
                            a, b, product = relevant_edge

                            # Add equation
                            # for k, v in edge_from_perspective_to_index:
                            #     print(k, v)
                            # print()

                            if rec_id == a: # a is simple machine
                                multi_idx = self.edge_from_perspective_to_index[(relevant_edge, b)]
                                # print(relevant_edge, rec_id, flush=True)
                                self.system.append(
                                    self.variables[self.arrayIndex(a, product, 'O')]
                                    -
                                    self.variables[self.arrayIndex(b, product, 'I', multi_idx=multi_idx)]
                                )
                            elif rec_id == b: # b is simple machine
                                multi_idx = self.edge_from_perspective_to_index[(relevant_edge, a)]
                                self.system.append(
                                    self.variables[self.arrayIndex(a, product, 'O', multi_idx=multi_idx)]
                                    -
                                    self.variables[self.arrayIndex(b, product, 'I')]
                                )

                computed_edges.update(involved_edges)


    def _addMultiEquationsOnEdge(self, multi_edge, multi_machine, destinations):
        # destinations = list of nodes

        multi_a, multi_b, multi_product = multi_edge
        if multi_machine == multi_a:
            direction = 'O'
        elif multi_machine == multi_b:
            direction = 'I'

        # Add new variables vX, vY, ...
        new_symbols = ', '.join(['v' + str(x + self.num_variables) for x in range(len(destinations))])
        self.variables.extend(list(symbols(new_symbols, positive=True, real=True)))

        # Look up old variable association with edge
        old_var = self.edge_from_perspective_to_index[(multi_edge, multi_machine)]
        # print(old_var)

        # Check for existing edge equations involving old_var in system and remove if exist
        # TODO: (ignoring this for now because it's not relevant for _addMachineMachineEdges)
        #   Consider keeping the machine-internal equation - it will remain accurate as the old var is kept for it

        # Update efpti and arrayIndex for new variables + edges
        connected_edges = self.graph.adj[multi_machine][direction]
        variable_index = self.num_variables
        for edge in connected_edges:
            a, b, product = edge
            if product != multi_product:
                continue
            
            self.edge_from_perspective_to_index[(edge, multi_machine)] = variable_index
            self.arrayIndex(multi_machine, product, direction, multi_idx=variable_index) # Sanity check that variable counts match
            variable_index += 1
        
        if direction == 'O':
            self.graph.parent_context.cLog(f'Solving multi-output scenario involving {multi_product}!', 'green', level=logging.INFO)
        elif direction == 'I':
            self.graph.parent_context.cLog(f'Solving multi-input scenario involving {multi_product}!', 'green', level=logging.INFO)

        # Add new equations for multi-IO
        # print(self.variables)
        # for k,v in self.lookup.items():
        #     print(k, v)

        base = self.variables[self.arrayIndex(multi_machine, multi_product, direction, multi_idx=0)]
        for i, dst in enumerate(destinations):
            base -= self.variables[self.arrayIndex(multi_machine, multi_product, direction, multi_idx=self.num_variables + i)]
        self.system.append(base)

        self.num_variables += len(destinations)


    def _solve(self):
        while True: # Loop until solved - algorithm may adjust edges each time it sees an EmptySet
            res = linsolve(self.system, self.variables)
            print(res)
            if isinstance(res, EmptySet):
                self._searchForInconsistency()
                exit(1)
            else:
                break

        lstres = list(res)
        if len(lstres) > 1:
            raise NotImplementedError('Multiple solutions - no code written to deal with this scenario yet')
        self.solved_vars = res.args[0]


    def _searchForInconsistency(self):
        # Solve each equation stepwise until inconsistency is found, then report to end user

        self.graph.parent_context.cLog('Searching for inconsistency in system of equations...', 'blue', level=logging.INFO)

        # for expr in system:
        #     print(expr)
        # print()

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

            if len(unsolved) <= 1:
                for var in solved:
                    expr = expr.subs(var, solved_values[var])
                # print('   ', expr)
                
                # If expr is a nonzero constant, inconsistency is found
                if expr.is_constant():
                    constant_diff = float(expr)
                    if not isclose(constant_diff, 0.0, abs_tol=0.00000001):
                        # self.graph.parent_context.cLog(f'Inconsistency found in {preexpr}!', 'red', level=logging.WARNING)
                        # Now print what the variables are referring to and output debug graph
                        inconsistent_variables.append((involved_variables, constant_diff))
                        iterations += 1
                        continue

                unvar = unsolved[0]
                solution = solve(expr, unvar)

                if len(unsolved) == 1:
                    if len(solution) == 1:
                        sval = solution[0]
                        solved_values[unvar] = sval
                        # print(sval)
                    else:
                        raise NotImplementedError(f'{solution=}')
                else:
                    raise NotImplementedError(f'{expr} {sorted(solved_values.items(), key=lambda tup: str(tup[0]))}')

            else:
                equations_to_check.append(expr)
            
            iterations += 1
        
        if inconsistent_variables == []:
            raise NotImplementedError('Both linear and nonlinear solver found empty set, so system of equations has no solutions -- report to dev.')

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
                self.graph.parent_context.cLog(f'Major inconsistency: {group}', 'red', level=logging.WARNING)

                self.graph.parent_context.cLog(f'Between output={self.graph.recipes[mpdm_cache[0][0]].O}', 'red', level=logging.WARNING)
                self.graph.parent_context.cLog(f'    and  input={self.graph.recipes[mpdm_cache[1][0]].I}', 'red', level=logging.WARNING)

                self.graph.parent_context.cLog('Please fix by either:', 'green', level=logging.INFO)

                # if constant_diff < 0:
                #     parent_group_idx = 0
                #     child_group_idx = 1
                # else:
                parent_group_idx = 0
                child_group_idx = 1

                # Negative means too much of right side, or too few of other sided inputs
                self.graph.parent_context.cLog(f'1. Sending excess {group[parent_group_idx]} {product} to sink', 'blue', level=logging.INFO)

                # Check other sided inputs
                machine, product, direction, multi_idx = idx_to_mpdm[var_to_idx(group[child_group_idx])]
                nonself_product = []
                for edge in self.graph.adj[machine][direction]:
                    # print(self.graph.adj[machine])
                    # print(edge)
                    a, b, edgeproduct = edge
                    if edgeproduct != product:
                        nonself_product.append((
                            edgeproduct,
                            'v' + f'{self.edge_from_perspective_to_index[(edge, machine)]}',
                        ))

                self.graph.parent_context.cLog(f'2. Pulling more {nonself_product} from source', 'blue', level=logging.INFO)

                # Output graph for end user to view
                self._debugAddVarsToEdges()
                self.graph.outputGraphviz()

                # TODO: Automate solution process fully

                selection = input() # TODO: Verify input

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
        #   def arrayIndex(machine, product, direction):
        # Edges in self.edges are defined like:
        #   rec_id_a, rec_id_b, product

        for edge_perspective_data, variableIndex in self.edge_from_perspective_to_index.items():
            edge, perspective = edge_perspective_data
            a, b, product = edge
            
            if 'debugHead' not in self.graph.edges[edge]:
                self.graph.edges[edge]['debugHead'] = ''
            if 'debugTail' not in self.graph.edges[edge]:
                self.graph.edges[edge]['debugTail'] = ''

            if perspective == b:
                self.graph.edges[edge]['debugHead'] += f'v{variableIndex}'
            elif perspective == a:
                self.graph.edges[edge]['debugTail'] += f'v{variableIndex}'
        
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



def addMachineMultipliers(self):
    # Compute machine multiplier based on solved ingredient quantities
    # FIXME: If multipliers disagree, sympy solver might have failed on an earlier step
    
    for rec_id, rec in self.recipes.items():
        multipliers = []

        for io_dir in ['I', 'O']:
            for ing in getattr(rec, io_dir):
                ing_name = ing.name
                base_quant = ing.quant

                # Look up edge value from sympy solver
                solved_quant_per_s = 0
                for edge in self.adj[rec_id][io_dir]:
                    if edge[2] == ing_name:
                        # print(edge, self.edges[edge]['quant'])
                        solved_quant_per_s += self.edges[edge]['quant']

                base_quant_s = base_quant / (rec.dur/20)
                
                # print(io_dir, rec_id, ing_name, getattr(rec, io_dir))
                # print(solved_quant_per_s, base_quant_s, rec.dur)
                # print()

                machine_multiplier = solved_quant_per_s / base_quant_s
                multipliers.append(machine_multiplier)

        final_multiplier = max(multipliers)
        rec.multiplier = final_multiplier
        rec.eut = rec.multiplier * rec.eut


def capitalizeMachine(machine):
    # check if machine has capitals, and if so, preserve them
    capitals = set(ascii_uppercase)
    machine_capitals = [ch for ch in machine if ch in capitals]

    capitalization_exceptions = {

    }
    
    if len(machine_capitals) > 0:
        return machine
    elif machine in capitalization_exceptions:
        return capitalization_exceptions[machine]
    else:
        return machine.title()


def createMachineLabels(self):
    # Distillation Tower
    # ->
    # 5.71x HV Distillation Tower
    # Cycle: 2.0s
    # Amoritized: 1.46K EU/t
    # Per Machine: 256EU/t
    
    for node_id in self.nodes:
        if self._checkIfMachine(node_id):
            rec_id = node_id
            rec = self.recipes[rec_id]
        else:
            continue

        label_lines = []

        # Standard label
        label_lines.extend([
            f'{round(rec.multiplier, 2)}x {rec.user_voltage.upper()} {capitalizeMachine(rec.machine)}',
            f'Cycle: {rec.dur/20}s',
            f'Amoritized: {self.userRound(int(round(rec.eut, 0)))} EU/t',
            f'Per Machine: {self.userRound(int(round(rec.base_eut, 0)))} EU/t',
        ])

        # Edits for power machines
        recognized_basic_power_machines = {
            # "basic" here means "doesn't cost energy to run"
            'gas turbine',
            'combustion gen',
            'semifluid gen',
            'steam turbine',
            'rocket engine',

            'large naquadah reactor',
            'large gas turbine',
            'large steam turbine',
            'large combustion engine',
            'extreme combustion engine',
            'XL Turbo Gas Turbine',
            'XL Turbo Steam Turbine',

            'air intake hatch',
        }
        if rec.machine in recognized_basic_power_machines:
            # Remove power input data
            label_lines = label_lines[:-2]
        
        line_if_attr_exists = {
            'heat': (lambda rec: f'Base Heat: {rec.heat}K'),
            'coils': (lambda rec: f'Coils: {rec.coils.title()}'),
            'saw_type': (lambda rec: f'Saw Type: {rec.saw_type.title()}'),
            'material': (lambda rec: f'Turbine Material: {rec.material.title()}'),
            'size': (lambda rec: f'Size: {rec.size.title()}'),
            'efficiency': (lambda rec: f'Efficiency: {rec.efficiency}'),
            'wasted_fuel': (lambda rec: f'Wasted Fuel: {rec.wasted_fuel}'),
        }
        for lookup, line_generator in line_if_attr_exists.items():
            if hasattr(rec, lookup):
                label_lines.append(line_generator(rec))

        self.nodes[rec_id]['label'] = '\n'.join(label_lines)


def addPowerLineNodesV2(self):
    generator_names = {
        0: 'gas turbine',
        1: 'combustion gen',
        2: 'semifluid gen',
        3: 'steam turbine',
        4: 'rocket engine',
        5: 'large naquadah reactor',
    }

    with open('data/power_data.yaml', 'r') as f:
        power_data = yaml.safe_load(f)
    with open('data/overclock_data.yaml', 'r') as f:
        overclock_data = yaml.safe_load(f)

    turbineables = power_data['turbine_fuels']
    combustables = power_data['combustion_fuels']
    semifluids = power_data['semifluids']
    rocket_fuels = power_data['rocket_fuels']
    naqline_fuels = power_data['naqline_fuels']

    known_burnables = {x: [0, y] for x,y in turbineables.items()}
    known_burnables.update({x: [1, y] for x,y in combustables.items()})
    known_burnables.update({x: [2, y] for x,y in semifluids.items()})
    known_burnables['steam'] = [3, 500]
    known_burnables.update({x: [4, y] for x,y in rocket_fuels.items()})
    known_burnables.update({x: [5, y] for x,y in naqline_fuels.items()})

    # Add new burn machines to graph - they will be computed for using new solver
    # 1. Find highest voltage on the chart - use this for burn generator tier
    # 2. Figure out highest node index on the chart - use this for adding generator nodes
    voltages = overclock_data['voltage_data']['tiers']
    highest_voltage = 0
    highest_node_index = 0
    for rec_id, rec in self.recipes.items():

        rec_voltage = voltages.index(rec.user_voltage)
        if rec_voltage > highest_voltage:
            highest_voltage = rec_voltage

        int_index = int(rec_id)
        if int_index > highest_node_index:
            highest_node_index = int_index

    highest_node_index += 1

    # 3. Redirect burnables currently going to sink and redirect them to a new burn machine
    outputs = self.adj['sink']['I']
    for edge in deepcopy(outputs):
        node_from, _, ing_name = edge
        edge_data = self.edges[edge]
        quant_s = edge_data['quant']

        if ing_name in known_burnables and not ing_name in self.graph_config['DO_NOT_BURN']:
            self.parent_context.cLog(f'Detected burnable: {ing_name.title()}! Adding to chart.', 'blue', level=logging.INFO)
            generator_idx, eut_per_cell = known_burnables[ing_name]
            gen_name = generator_names[generator_idx]

            # Add node
            node_idx = f'{highest_node_index}'

            # Burn gen is a singleblock
            def findClosestVoltage(voltage_list, voltage):
                nonlocal voltages
                leftmost = voltages.index(voltage_list[0])
                rightmost = voltages.index(voltage_list[-1])
                target = voltages.index(voltage)

                # First try to voltage down
                if rightmost < target:
                    return voltages[rightmost]
                elif leftmost <= target <= rightmost:
                    return voltages[target]
                elif leftmost > target:
                    return voltages[leftmost]

            available_efficiencies = power_data['simple_generator_efficiencies'][gen_name]
            gen_voltage = findClosestVoltage(list(available_efficiencies), voltages[highest_voltage])
            efficiency = available_efficiencies[gen_voltage]

            # Compute I/O for a single tick
            gen_voltage_index = voltages.index(gen_voltage)
            output_eut = 32 * (4 ** gen_voltage_index)
            loss_on_singleblock_output = (2 ** (gen_voltage_index+1))
            expended_eut = output_eut + loss_on_singleblock_output

            expended_fuel_t = expended_eut / (eut_per_cell/1000 * efficiency)

            gen_input = IngredientCollection(
                Ingredient(
                    ing_name,
                    expended_fuel_t
                )
            )
            gen_output = IngredientCollection(
                Ingredient(
                    'EU',
                    output_eut
                )
            )

            # Append to recipes
            self.recipes[str(highest_node_index)] = Recipe(
                gen_name,
                gen_voltage,
                gen_input,
                gen_output,
                0,
                1,
                efficiency=f'{efficiency*100}%',
                wasted_fuel=f'{self.userRound(loss_on_singleblock_output)}EU/t/amp',
            )

            produced_eut_s = quant_s/expended_fuel_t*output_eut 
            self.parent_context.cLog(
                ''.join([
                    f'Added {gen_voltage} generator burning {quant_s} {ing_name} for '
                    f'{self.userRound(produced_eut_s/20)}EU/t at {output_eut}EU/t each.'
                ]),
                'blue',
                level=logging.INFO,
            )

            self.addNode(
                node_idx,
                fillcolor=self.graph_config['NONLOCKEDNODE_COLOR'],
                shape='box'
            )

            # Fix edges to point at said node
            # Edge (old output) -> (generator)
            self.addEdge(
                node_from,
                node_idx,
                ing_name,
                quant_s,
                **edge_data['kwargs'],
            )
            # Edge (generator) -> (EU sink)
            self.addEdge(
                node_idx,
                'sink',
                'EU',
                produced_eut_s,
            )
            # Remove old edge and repopulate adjacency list
            del self.edges[edge]
            self.createAdjacencyList()

            highest_node_index += 1


def addUserNodeColor(self):
    targeted_nodes = [i for i, x in self.recipes.items() if getattr(x, 'target', False) != False]
    numbered_nodes = [i for i, x in self.recipes.items() if getattr(x, 'number', False) != False]
    all_user_nodes = set(targeted_nodes) | set(numbered_nodes)

    for rec_id in all_user_nodes:
        self.nodes[rec_id].update({'fillcolor': self.graph_config['LOCKEDNODE_COLOR']})


def graphPreProcessing(self):
    self.connectGraph()
    self.removeBackEdges()
    self.createAdjacencyList()


def graphPostProcessing(self):
    if self.graph_config.get('POWER_LINE', False):
        addPowerLineNodesV2(self)

    addMachineMultipliers(self)
    createMachineLabels(self)

    self._addSummaryNode()

    addUserNodeColor(self)

    if self.graph_config.get('COMBINE_INPUTS', False):
        self._combineInputs()
    if self.graph_config.get('COMBINE_OUTPUTS', False):
        self._combineOutputs()


def systemOfEquationsSolverGraphGen(self, project_name, recipes, graph_config):
    g = Graph(project_name, recipes, self, graph_config=graph_config)
    graphPreProcessing(g)

    g.parent_context.cLog('Running linear solver...', 'green', level=logging.INFO)
    solver = SympySolver(g)
    solver.run()

    graphPostProcessing(g)
    g.outputGraphviz()


if __name__ == '__main__':
    from factory_graph import ProgramContext
    c = ProgramContext()

    c.run(graph_gen=systemOfEquationsSolverGraphGen)