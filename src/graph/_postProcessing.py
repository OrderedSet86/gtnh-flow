import math
from collections import defaultdict
from copy import deepcopy
from string import ascii_uppercase
from typing import Generator, Union

import yaml
from termcolor import colored, cprint

from src.data.basicTypes import Ingredient, IngredientCollection, Recipe
from src.graph._utils import _iterateOverMachines
from src.gtnh.overclocks import OverclockHandler



def capitalizeMachine(machine: str) -> str:
    # check if machine has capitals, and if so, preserve them
    capitals = set(ascii_uppercase)
    machine_capitals = [ch for ch in machine if ch in capitals]

    capitalization_exceptions = {
        # Format is old_str: new_str
    }

    if len(machine_capitals) > 0:
        return machine
    elif machine in capitalization_exceptions:
        return capitalization_exceptions[machine]
    else:
        return machine.title()



def createMachineLabels(self) -> None:
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
            f'Cycle: {round(rec.dur/20, 2)}s',
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
            'nc': (lambda rec: f'NC: {rec.nc.title()}'),
            'heat': (lambda rec: f'Base Heat: {rec.heat}K'),
            'coils': (lambda rec: f'Coils: {rec.coils.title()}'),
            'saw_type': (lambda rec: f'Saw Type: {rec.saw_type.title()}'),
            'material': (lambda rec: f'Turbine Material: {rec.material.title()}'),
            'size': (lambda rec: f'Size: {rec.size.title()}'),
            'efficiency': (lambda rec: f'Efficiency: {rec.efficiency}'),
            'wasted_fuel': (lambda rec: f'Wasted Fuel: {rec.wasted_fuel}'),
            'parallel': (lambda rec: f'Parallels: {rec.parallel}'),
            'note': (lambda rec: f'Note: {rec.note}'),
            'pipe_casings': (lambda rec: f'Pipe Casings: {rec.pipe_casings}'),
        }
        for lookup, line_generator in line_if_attr_exists.items():
            if hasattr(rec, lookup):
                label_lines.append(line_generator(rec))

        self.nodes[rec_id]['label'] = '\n'.join(label_lines)



def addUserNodeColor(self) -> None:
    targeted_nodes = [i for i, x in self.recipes.items() if getattr(x, 'target', False) != False]
    numbered_nodes = [i for i, x in self.recipes.items() if getattr(x, 'number', False) != False]
    all_user_nodes = set(targeted_nodes) | set(numbered_nodes)

    for rec_id in all_user_nodes:
        # Emphasizes the first line (i.e. machine name line)
        lines = self.nodes[rec_id]['label'].split('\n')
        lines[0] = r'<b><u>' + lines[0] + r'</u></b>'
        self.nodes[rec_id]['label'] = '\n'.join(lines)


def addMachineMultipliers(self) -> None:
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


def addPowerLineNodesV2(self) -> None:
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
    burn_machines_added = False
    for edge in deepcopy(outputs):
        node_from, _, ing_name = edge
        edge_data = self.edges[edge]
        quant_s = edge_data['quant']

        if ing_name in known_burnables and not ing_name in self.graph_config['DO_NOT_BURN']:
            self.parent_context.log.info(colored(f'Detected burnable: {ing_name.title()}! Adding to chart.', 'blue'))
            burn_machines_added = True

            generator_idx, eut_per_cell = known_burnables[ing_name]
            gen_name = generator_names[generator_idx]

            # Add node
            node_idx = f'{highest_node_index}'

            # Burn gen is a singleblock
            def findClosestVoltage(voltage_list: list[str], voltage: str) -> str:
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
            loss_on_singleblock_output = (2 ** (gen_voltage_index+1-1))
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
            self.parent_context.log.info(
                colored(
                    ''.join([
                        f'Added {gen_voltage} generator burning {quant_s} {ing_name} for '
                        f'{self.userRound(produced_eut_s/20)}EU/t at {output_eut}EU/t each.'
                    ]),
                    'blue',
                )
            )

            self.addNode(
                node_idx,
                fillcolor=self.graph_config['DEFAULT_MACHINE_COLOR'],
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

            highest_node_index += 1

    # 4. Special UCFE handling
    ### Automatically balance the outputs of UCFE
    # 1. Get UCFE node
    UCFE_id = None
    for rec_id, rec in self.recipes.items():
        if rec.machine == 'universal chemical fuel engine':
            UCFE_id = rec_id

    if UCFE_id is not None:
        cprint('Detected UCFE, autobalancing...', 'green')

        # 2. Determine whether non-combustion promoter input is combustable or gas
        input_ingredient_collection = self.recipes[UCFE_id].I
        if len(input_ingredient_collection) != 2:
            raise RuntimeError('Too many or too few inputs to UCFE - expected 2.')

        if 'combustion promoter' not in input_ingredient_collection._ingdict:
            raise RuntimeError('UCFE detected, but "combustion promoter" is not one of its inputs. Cannot autobalance.')

        for ing in input_ingredient_collection._ings:
            if ing.name != 'combustion promoter':
                fuel_name = ing.name
                break

        burn_value_table = None
        if fuel_name in turbineables:
            burn_value_table = turbineables
            coefficient = 0.04
        elif fuel_name in combustables:
            burn_value_table = combustables
            coefficient = 0.04
        elif fuel_name in rocket_fuels:
            burn_value_table = rocket_fuels
            coefficient = 0.005
        else:
            raise RuntimeError(f'Unrecognized input fuel to UCFE: {fuel_name}. Can only burn gas, combustables, or rocket fuel.')

        # 3. Compute UCFE ratio and output EU/s
        combustion_promoter_quant = input_ingredient_collection['combustion promoter'][0]
        fuel_quant = input_ingredient_collection[fuel_name][0]
        ratio = fuel_quant / combustion_promoter_quant

        efficiency = math.exp(-coefficient*ratio) * 1.5
        output_eu = efficiency * fuel_quant * burn_value_table[fuel_name] / 1000
        print(f'UCFE ratio: {ratio}, efficiency: {efficiency}, output EU/s: {output_eu}')

        # 4. Update edge with new value
        self.edges[(UCFE_id, 'sink', 'EU')]['quant'] = output_eu

        # 5. Fix insane multiplier and label numbers
        UCFE_rec = self.recipes[UCFE_id]
        UCFE_rec.multiplier = 1
        UCFE_rec.O = IngredientCollection(Ingredient('EU', output_eu))

    if burn_machines_added:
        self.parent_context.log.debug(colored('Updating adj since new powerline machines added', 'yellow'))
        self.createAdjacencyList()


def addSummaryNode(self) -> None:
    # Now that tree is fully locked, add I/O node
    # Specifically, inputs are adj[source] and outputs are adj[sink]
    with open('data/misc.yaml', 'r') as f:
        misc_data = yaml.safe_load(f)
    with open('data/overclock_data.yaml', 'r') as f:
        overclock_data = yaml.safe_load(f)

    color_positive = self.graph_config['POSITIVE_COLOR']
    color_negative = self.graph_config['NEGATIVE_COLOR']

    def makeLineHtml(
            lab_text: str,
            amt_text: str,
            lab_color: str,
            amt_color: str,
        ) -> str:
        nonlocal self
        return ''.join([
            '<tr>'
            f'<td align="left"><font color="{lab_color}" face="{self.graph_config["SUMMARY_FONT"]}">{self.stripBrackets(lab_text)}</font></td>'
            f'<td align ="right"><font color="{amt_color}" face="{self.graph_config["SUMMARY_FONT"]}">{amt_text}</font></td>'
            '</tr>'
        ])

    self.parent_context.log.debug(colored('Updating adj before summary node', 'yellow'))
    self.createAdjacencyList()

    # Compute I/O
    total_io = defaultdict(float)
    ing_names = defaultdict(str)
    input_flows = defaultdict(float)
    for direction in [-1, 1]:
        if direction == -1:
            # Inputs
            edges = self.adj['source']['O']
        elif direction == 1:
            # Outputs
            edges = self.adj['sink']['I']

        for edge in edges:
            _, _, ing_name = edge
            edge_data = self.edges[edge]
            quant = edge_data['quant']

            ing_id = self.getIngId(ing_name)

            ing_names[ing_id] = self.getIngLabel(ing_name)
            total_io[ing_id] += direction * quant
            if direction == -1:
                input_flows[ing_id] += direction * quant

    def canonicalizeFlow(flow: float) -> Union[int, float]:
        # Set to 0 if too small (intended to avoid floating point issues)
        return flow if abs(flow) > 1e-5 else 0
    total_io = {ing: canonicalizeFlow(flow) for ing, flow in total_io.items()}

    # Create I/O lines
    io_label_lines = []

    def makeIOTitle(title: str) -> str:
        font = self.graph_config["SUMMARY_FONT"]
        return f'<tr><td align="left"><font color="white" face="{font}"><b>{title}</b></font></td></tr><hr/>'

    def makeIOLines(flows: list[float], color: str) -> Generator[str, None, None]:
        for id, quant in sorted(flows, key=lambda x: -abs(x[1])):
            amt_text = self.getQuantLabel(id, quant)
            name_text = '\u2588 ' + ing_names[id]
            num_color = color
            ing_color = self.getUniqueColor(id)
            yield makeLineHtml(name_text, amt_text, ing_color, num_color)

    ## If one ingredient's net output is equal or greater than 0, it is recyclable
    recyclable_flows = {id: quant for id, quant in total_io.items() if id != 'eu' and input_flows.get(id, 0) < 0 and total_io[id] >= 0}
    color_recyclable = self.graph_config['RECYCLABLE_COLOR']
    io_label_lines.append(makeIOTitle('Input'))
    io_label_lines.extend(makeIOLines(
        filter(lambda e: e[0] != 'eu' and e[0] not in recyclable_flows and e[1] < 0, total_io.items()),
        color_negative
    ))
    io_label_lines.append(makeIOTitle('Output'))
    io_label_lines.extend(makeIOLines(
        filter(lambda e: e[0] != 'eu' and e[0] not in recyclable_flows and e[1] > 0, total_io.items()),
        color_positive
    ))
    ## There might not be recyclable inputs
    if recyclable_flows:
        io_label_lines.append(makeIOTitle('Recyclable'))
        io_label_lines.extend(makeIOLines(recyclable_flows.items(), color_recyclable))

    # Compute total EU/t cost and (if power line) output
    total_eut = 0
    for rec in self.recipes.values():
        total_eut += rec.eut
    if io_label_lines[-1] == '':
        io_label_lines.append('<hr/>')
    eut_rounded = -int(math.ceil(total_eut))
    io_label_lines.append(makeLineHtml('Input EU/t:', self.userRound(eut_rounded), 'white', color_negative))
    if 'eu' in total_io:
        produced_eut = int(math.floor(total_io['eu'] / 20))
        io_label_lines.append(makeLineHtml('Output EU/t:', self.userRound(produced_eut), 'white', color_positive))
        net_eut = produced_eut + eut_rounded
        lab_color = 'white'
        amt_color = color_positive if net_eut >= 0 else color_negative
        io_label_lines.append(makeLineHtml('Net EU/t:', self.userRound(net_eut), lab_color, amt_color))
        io_label_lines.append('<hr/>')

    # Add total machine multiplier count for renewables spreadsheet numbers
    special_machine_weights = misc_data['special_machine_weights']
    sumval = 0
    for rec_id in self.nodes:
        if rec_id in ['source', 'sink']:
            continue
        elif rec_id.startswith('power_'):
            continue
        rec = self.recipes[rec_id]

        machine_weight = rec.multiplier
        if rec.machine in special_machine_weights:
            machine_weight *= special_machine_weights[rec.machine]
        sumval += machine_weight

    io_label_lines.append(makeLineHtml('Total machine count:', self.userRound(sumval), 'white', color_positive))

    # Add peak power load in maximum voltage on chart
    # Find maximum voltage
    max_tier = -1
    tiers = overclock_data['voltage_data']['tiers']
    for rec in self.recipes.values():
        tier = tiers.index(rec.user_voltage)
        if tier > max_tier:
            max_tier = tier
    voltage_at_tier = 32 * pow(4, max_tier)

    # Compute maximum draw
    max_draw = 0
    for rec in self.recipes.values():
        max_draw += rec.base_eut * math.ceil(rec.multiplier)

    io_label_lines.append(
        makeLineHtml(
            'Peak power draw:',
            f'{round(max_draw/voltage_at_tier, 2)}A {tiers[max_tier].upper()}',
            'white',
            color_negative
        )
    )

    # Create final table
    io_label = ''.join(io_label_lines)
    io_label = f'<<table border="0">{io_label}</table>>'

    # Add to graph
    self.addNode(
        'total_io_node',
        label=io_label,
        fillcolor=self.graph_config['BACKGROUND_COLOR'],
        shape='box'
    )


def bottleneckPrint(self) -> None:
    # Prints bottlenecks normalized to an input voltage.
    machine_recipes = [x for x in _iterateOverMachines(self)]
    machine_recipes.sort(
        key=lambda rec: rec.multiplier,
        reverse=True,
    )

    max_print = self.graph_config.get('MAX_BOTTLENECKS')
    number_to_print = max(len(machine_recipes)//10, max_print)

    if self.graph_config.get('USE_BOTTLENECK_EXACT_VOLTAGE'):
        # Want to overclock and underclock to force the specific voltage
        chosen_voltage = self.graph_config.get('BOTTLENECK_MIN_VOLTAGE')

        oh = OverclockHandler(self.parent_context)
        raise NotImplementedError() # FIXME: Add negative overclocking
        for i, rec in enumerate(self.recipes):
            rec.user_voltage = chosen_voltage

            self.recipes[i] = oh.overclockRecipe(rec)
            rec.base_eut = rec.eut

    # Print actual bottlenecks
    for i, rec in zip(range(number_to_print), machine_recipes):
        cprint(f'{round(rec.multiplier, 2)}x {rec.user_voltage} {rec.machine}', 'red')
        for out in rec.O:
            cprint(f'    {out.name.title()} ({round(out.quant, 2)})', 'green')
