# In theory solving the machine flow as a linear program is fast and simple -
# this prototype explores this.

import logging
from copy import deepcopy
from math import isclose
from string import ascii_uppercase

import yaml
from sympy import linsolve, symbols

from src.data.basicTypes import Ingredient, IngredientCollection, Recipe
from src.graph import Graph


def sympySolver(self):
    # System of equations:
    # input_per_s_1 = machine_3_count * output_per_s_3 + machine_5_count * output_per_s_5
    # output_per_s_1 = (constant) * input_per_s_1
    # ...

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
            solved_quant_s = core_ing.quant * rec.number / (rec.dur/20)
            system.append(
                variables[arrayIndex(rec_id, core_ing.name, core_direction)]
                -
                solved_quant_s
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
            # print(f'Machine edge detected! {edge}')
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

    # for machine_info, index in lookup.items():
    #     print(machine_info, solved_vars[index])

    # Update graph edge values
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

            if isclose(a_quant, b_quant, rel_tol=0.05):
                relevant_edge = self.edges[edge]
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
                solution_index = lookup[(a, product, 'O')]
            elif b_machine:
                solution_index = lookup[(b, product, 'I')]

            quant = solved_vars[solution_index]
            relevant_edge = self.edges[edge]
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
                for edge in self.adj_machine[rec_id][io_dir]:
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

            node_name = f'{gen_name.title()} ({int(efficiency*100)}% eff)'

            # Compute I/O for a single tick
            gen_voltage_index = voltages.index(gen_voltage)
            output_eut = 32 * (4 ** gen_voltage_index)
            loss_on_singleblock_output = (2 ** gen_voltage_index)
            expended_eut = output_eut + loss_on_singleblock_output

            expended_fuel_t = (eut_per_cell/1000 * efficiency) / expended_eut

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
                label= node_name,
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

    if self.graph_config.get('COMBINE_INPUTS', False):
        self._combineInputs()
    if self.graph_config.get('COMBINE_OUTPUTS', False):
        self._combineOutputs()


def systemOfEquationsSolverGraphGen(self, project_name, recipes, graph_config):
    g = Graph(project_name, recipes, self, graph_config=graph_config)

    graphPreProcessing(g)
    sympySolver(g)
    graphPostProcessing(g)

    g.outputGraphviz()


if __name__ == '__main__':
    from factory_graph import ProgramContext
    c = ProgramContext()

    c.run(graph_gen=systemOfEquationsSolverGraphGen)