import logging
import math
import re
from collections import defaultdict
from copy import deepcopy


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
    turbineables = {
        'hydrogen': 20_000,
        'natural gas': 20_000,
        'carbon monoxide': 24_000,
        'wood gas': 24_000,
        'sulfuric gas': 25_000,
        'biogas': 40_000,
        'sulfuric naphtha': 40_000,
        'cyclopentadiene': 70_000,
        'coal gas': 96_000,
        'methane': 104_000,
        'ethylene': 128_000,
        'refinery gas': 160_000,
        'ethane': 168_000,
        'propene': 192_000,
        'butadiene': 206_000,
        'propane': 232_000,
        'rocket fuel': 250_000,
        'butene': 256_000,
        'phenol': 288_000,
        'benzene': 360_000,
        'butane': 296_000,
        'lpg': 320_000,
        'naphtha': 320_000,
        'toluene': 328_000,
        'tert-butylbenzene': 420_000,
        'naquadah gas': 1_024_000,
        'nitrobenzene': 1_250_000,
    }
    combustables = {
        'fish oil': 2_000,
        'short mead': 4_000,
        'biofuel': 6_000,
        'creosote oil': 8_000,
        'biomass': 8_000,
        'oil': 16_000,
        'sulfuric light fuel': 40_000,
        'octane': 80_000,
        'methanol': 84_000,
        'ethanol': 192_000,
        'bio diesel': 320_000,
        'light fuel': 305_000,
        'diesel': 480_000,
        'ether': 537_000,
        'gasoline': 576_000,
        'cetane-boosted diesel': 1_000_000,
        'ethanol gasoline': 750_000,
        'butanol': 1_125_000,
        'jet fuel no.3': 1_324_000,
        'high octane gasoline': 2_500_000,
        'jet fuel A': 2_048_000,
    }
    semifluids = {
        'seed oil': 4_000,
        'fish oil': 4_000,
        'raw animal waste': 12_000,
        'biomass': 16_000,
        'coal tar': 16_000,
        'manure slurry': 24_000,
        'coal tar oil': 32_000,
        'fertile manure slurry': 32_000,
        'oil': 40_000,
        'light oil': 40_000,
        'creosote oil': 48_000,
        'raw oil': 60_000,
        'heavy oil': 60_000,
        'sulfuric coal tar oil': 64_000,
        'sulfuric heavy fuel': 80_000,
        'heavy fuel': 360_000,
    }
    rocket_fuels = {
        'rp-1 rocket fuel': 1_536_000,
        'lmp-103s': 1_998_000,
        'dense hydrazine fuel mixture': 3_072_000,
        'monomethylhydrazine fuel mix': 4_500_000,
        'cn3h7o3 rocket fuel': 6_144_000,
        'unsymmetrical dimethylhydrazine fuel mix': 9_000_000,
        'h8n4c2o4 rocket fuel': 12_588_000,
    }
    naqline_fuels = {
        'naquadah based liquid fuel mkI': 220_000*20*1000,
        'naquadah based liquid fuel mkII': 380_000*20*1000,
        'naquadah based liquid fuel mkIII': 9_511_000*80*1000,
        'naquadah based liquid fuel mkIV': 88_540_000*100*1000,
        'naquadah based liquid fuel mkV': 399_576_000*8*20*1000,
        'uranium based liquid fuel (excited state)': 12_960*100*1000,
        'plutonium based liquid fuel (excited state)': 32_400*7.5*20*1000,
        'thorium based liquid fuel (excited state)': 2_200*25*20*1000,
    }
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

    # Add total machine multiplier count for oxygen table
    sumval = 0
    special_machine_weights = {
        'distillation tower': 10,
        'pyrolyse oven': 5,
        'electric blast furnace': 5,
        'multi smelter': 3,
        'zhuhai': 3,
        'vacuum freezer': 3,
        'electric blast furnace': 5,
    }
    for rec_id in self.nodes:
        if rec_id in ['source', 'sink']:
            continue
        elif re.match(r'^power_\d+_\d+$', rec_id):
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
    tiers = ['LV', 'MV', 'HV', 'EV', 'IV', 'LuV', 'ZPM', 'UV', 'UHV', 'UEV', 'UIV', 'UMV', 'UXV']
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
            f'{round(max_draw/voltage_at_tier, 2)}A {tiers[max_tier]}',
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