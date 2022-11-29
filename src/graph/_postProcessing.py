import logging
import math
import re
from collections import defaultdict
from copy import deepcopy

import yaml
from termcolor import colored, cprint

from src.graph._utils import _iterateOverMachines
from src.gtnh.overclocks import OverclockHandler


def _addPowerLineNodes(self):
    # This checks for burnables being put into sink and converts them to EU/t
    generator_names = {
        0: 'gas turbine',
        1: 'combustion gen',
        2: 'semifluid gen',
        3: 'steam turbine',
        4: 'rocket engine fuel',
        5: 'large naquadah reactor',
    }

    with open('data/power_data.yaml', 'r') as f:
        power_data = yaml.safe_load(f)

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

    outputs = self.adj['sink']['I']
    generator_number = 1
    for edge in deepcopy(outputs):
        node_from, _, ing_name = edge
        edge_data = self.edges[edge]
        quant = edge_data['quant']

        if ing_name in known_burnables and not ing_name in self.graph_config['DO_NOT_BURN']:
            self.parent_context.cLog(f'Detected burnable: {ing_name.title()}! Adding to chart.', 'blue', level=logging.INFO)
            generator_idx, eut_per_cell = known_burnables[ing_name]
            gen_name = generator_names[generator_idx].title()

            # Add node
            node_gen = f'power_{generator_number}_{generator_idx}'
            generator_number += 1
            node_name = f'{gen_name} (100% eff)'
            self.addNode(
                node_gen,
                label= node_name,
                fillcolor=self.graph_config['NONLOCKEDNODE_COLOR'],
                shape='box'
            )

            # Fix edges to point at said node
            produced_eut = eut_per_cell * quant / 1000
            # Edge (old output) -> (generator)
            self.addEdge(
                node_from,
                node_gen,
                ing_name,
                quant,
                **edge_data['kwargs'],
            )
            # Edge (generator) -> (EU sink)
            self.addEdge(
                node_gen,
                'sink',
                'EU',
                produced_eut,
            )
            # Remove old edge and repopulate adjacency list
            del self.edges[edge]
            self.createAdjacencyList()

    ### Automatically balance the outputs of UCFE
    # 1. Get UCFE node
    UCFE_id = None
    for rec_id, rec in self.recipes.items():
        if rec.machine == 'universal chemical fuel engine':
            UCFE_id = rec_id

    if UCFE_id is not None:
        self.parent_context.cLog('Detected UCFE, autobalancing...', 'green', level=logging.INFO)

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
        ratio = combustion_promoter_quant / fuel_quant
        self.parent_context.cLog(f'UCFE power ratio: {ratio}', 'green', level=logging.INFO)

        efficiency = math.exp(-coefficient*ratio) * 1.5
        self.parent_context.cLog(f'Efficiency stat: {efficiency}', 'green', level=logging.INFO)
        output_eu = efficiency * burn_value_table[fuel_name] * (fuel_quant / 1000)

        # 4. Update edge with new value
        self.edges[(UCFE_id, 'sink', 'EU')]['quant'] = output_eu


def _addSummaryNode(self):
    # Now that tree is fully locked, add I/O node
    # Specifically, inputs are adj[source] and outputs are adj[sink]
    with open('data/misc.yaml', 'r') as f:
        misc_data = yaml.safe_load(f)
    with open('data/overclock_data.yaml', 'r') as f:
        overclock_data = yaml.safe_load(f)

    color_positive = self.graph_config['POSITIVE_COLOR']
    color_negative = self.graph_config['NEGATIVE_COLOR']
    
    def makeLineHtml(lab_text, amt_text, lab_color, amt_color):
        return ''.join([
            '<tr>'
            f'<td align="left"><font color="{lab_color}" face="{self.graph_config["SUMMARY_FONT"]}">{self.stripBrackets(lab_text)}</font></td>'
            f'<td align ="right"><font color="{amt_color}" face="{self.graph_config["SUMMARY_FONT"]}">{amt_text}</font></td>'
            '</tr>'
        ])

    self.createAdjacencyList()

    # Compute I/O
    total_io = defaultdict(float)
    ing_names = defaultdict(str)
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

    # Create I/O lines
    io_label_lines = []
    io_label_lines.append(f'<tr><td align="left"><font color="white" face="{self.graph_config["SUMMARY_FONT"]}"><b>Summary</b></font></td></tr><hr/>')

    for id, quant in sorted(total_io.items(), key=lambda x: x[1]):
        if id == 'eu':
            continue

        # Skip if too small (intended to avoid floating point issues)
        near_zero_range = 10**-5
        if -near_zero_range < quant < near_zero_range:
            continue

        amt_text = self.getQuantLabel(id, quant)
        name_text = '\u2588 ' + ing_names[id]
        num_color = color_positive if quant >= 0 else color_negative
        ing_color = self.getUniqueColor(id)
        io_label_lines.append(makeLineHtml(name_text, amt_text, ing_color, num_color))

    # Compute total EU/t cost and (if power line) output
    total_eut = 0
    for rec in self.recipes.values():
        total_eut += rec.eut
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


def bottleneckPrint(self):
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
