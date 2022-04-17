# Graphs are traditionally collections of nodes and edges.
# However, recipes represent a node and all its connected edges.
# As a result, a special algorithm is required to connect all relevant edges in a factory.

# Standard libraries
import math
import itertools
from collections import defaultdict, OrderedDict
from copy import deepcopy

# Pypi libraries
import graphviz
from sigfig import round as sigfig_round
from termcolor import cprint

# Internal libraries
from dataClasses.base import Ingredient, IngredientCollection, Recipe
from graphClasses.backEdges import BasicGraph, dfs
from gtnhClasses.overclocks import overclockRecipe


def swapIO(io_type):
    if io_type == 'I':
        return 'O'
    elif io_type == 'O':
        return 'I'
    else:
        raise RuntimeError(f'Improper I/O type: {io_type}')

class Graph:
    def __init__(self, graph_name, recipes, graph_config=None):
        self.graph_name = graph_name
        self.recipes = {str(i): x for i, x in enumerate(recipes)}
        self.nodes = {}
        self.edges = {} # uniquely defined by (machine from, machine to, ing name)
        self.graph_config = graph_config
        if self.graph_config == None:
            self.graph_config = {}

        # Populated later on
        self.adj = None
        self.adj_machine = None

        self.darkModeColor = '#043742'

        # TODO: Temporary until backend data import
        for i, rec in enumerate(recipes):
            recipes[i] = overclockRecipe(rec)


    def addNode(self, recipe_id, **kwargs):
        self.nodes[recipe_id] = kwargs
    def addEdge(self, node_from, node_to, ing_name, quantity, **kwargs):
        self.edges[(node_from, node_to, ing_name)] = {
            'quant': quantity,
            'kwargs': kwargs
        }


    def connectGraph(self):
        '''
        Connects recipes without locking the quantities
        '''

        # Create source and sink nodes
        self.addNode('source', fillcolor='ghostwhite', label='source')
        self.addNode('sink', fillcolor='ghostwhite', label='sink')

        # Compute {[ingredient name][IO direction] -> involved recipes} table
        involved_recipes = defaultdict(lambda: defaultdict(list))
        for rec_id, rec in self.recipes.items():
            for io_type in ['I', 'O']:
                for ing in getattr(rec, io_type):
                    involved_recipes[ing.name][io_type].append(rec_id)

        # Add I/O connections
        added_edges = set()
        for rec_id, rec in self.recipes.items():
            # Create machine label
            if self.graph_config['SHOW_MACHINE_INDICES']:
                machine_label = [f'({rec_id}) {rec.machine.title()}']
            else:
                machine_label = [rec.machine.title()]

            # Add lines for the special arguments
            line_if_attr_exists = {
                'heat': (lambda rec: f'Base Heat: {rec.heat}K'),
                'coils': (lambda rec: f'Coils: {rec.coils.title()}'),
            }
            for lookup, line_generator in line_if_attr_exists.items():
                if hasattr(rec, lookup):
                    machine_label.append(line_generator(rec))

            machine_label = '\n'.join(machine_label)
            self.addNode(
                rec_id,
                fillcolor='lightblue2',
                label=machine_label
            )
            for io_type in ['I', 'O']:
                for ing in getattr(rec, io_type):
                    linked_machines = involved_recipes[ing.name][swapIO(io_type)]
                    if len(linked_machines) == 0:
                        if io_type == 'I':
                            linked_machines = ['source']
                        elif io_type == 'O':
                            linked_machines = ['sink']

                    for link_id in linked_machines:
                        # Skip already added edges
                        unique_edge_identifiers = [
                            (link_id, rec_id, ing.name),
                            (rec_id, link_id, ing.name)
                        ]
                        if any(x in added_edges for x in unique_edge_identifiers):
                            continue

                        if io_type == 'I':
                            self.addEdge(
                                str(link_id),
                                str(rec_id),
                                ing.name,
                                -1,
                            )
                            added_edges.add(unique_edge_identifiers[0])
                        elif io_type == 'O':
                            self.addEdge(
                                str(rec_id),
                                str(link_id),
                                ing.name,
                                -1,
                            )
                            added_edges.add(unique_edge_identifiers[1])

        if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
            self.outputGraphviz()


    def removeBackEdges(self):
        # Loops are possible in machine processing, but very difficult / NP-hard to solve properly
        # Want to make algorithm simple, so just break all back edges and send them to sink instead
        # The final I/O information will have these balanced, so this is ok

        # Run DFS back edges detector
        basic_edges = [(x[0], x[1]) for x in self.edges.keys()]
        G = BasicGraph(basic_edges)
        dfs(G)

        for back_edge in G.back_edges:
            # Note that although this doesn't include ingredient information, all edges between these two nodes
            # should be redirected
            from_node, to_node = back_edge
            relevant_edges = []
            for edge in self.edges.items():
                edge_def, edge_data = edge
                if (edge_def[0], edge_def[1]) == (from_node, to_node):
                    relevant_edges.append((edge_def, edge_data))

            for edge_def, edge_data in relevant_edges:
                node_from, node_to, ing_name = edge_def
                cprint(f'Fixing factory cycle by redirecting "{ing_name.title()}" to sink', 'yellow')

                # Redirect looped ingredient to sink
                self.addEdge(
                    node_from,
                    'sink',
                    ing_name,
                    edge_data['quant'],
                    **edge_data['kwargs']
                )
                # Pull newly required ingredients from source
                self.addEdge(
                    'source',
                    node_to,
                    ing_name,
                    edge_data['quant'],
                    **edge_data['kwargs']
                )

                del self.edges[edge_def]


    def createAdjacencyList(self):
        # Compute "adjacency list" (node -> {I: edges, O: edges}) for edges and machine-involved edges
        adj = defaultdict(lambda: defaultdict(list))
        adj_machine = defaultdict(lambda: defaultdict(list))
        for edge in self.edges:
            node_from, node_to, ing_name = edge
            adj[node_from]['O'].append(edge)
            adj[node_to]['I'].append(edge)
            if node_to not in {'sink', 'source'}:
                adj_machine[node_from]['O'].append(edge)
            if node_from not in {'sink', 'source'}:
                adj_machine[node_to]['I'].append(edge)

        self.adj = adj
        self.adj_machine = adj_machine

        # TODO: Add to debug print only
        for machine, io_group in self.adj_machine.items():
            machine_name = ''
            recipe_obj = self.recipes.get(machine)
            if isinstance(recipe_obj, Recipe):
                machine_name = recipe_obj.machine

            cprint(f'{machine} {machine_name}', 'blue')
            for io_type, edges in io_group.items():
                cprint(f'{io_type} {edges}', 'blue')
        print()


    def balanceGraph(self):
        # Applies locking info to existing graph
        self.removeBackEdges()

        # Create adjacency list for easier compute
        self.createAdjacencyList()
        adj = self.adj
        adj_machine = self.adj_machine

        # Debug
        for node, adj_edges in adj_machine.items():
            if node in ['sink', 'source']:
                continue
            print(node, dict(adj_edges))
        print()

        # Locking rules:
        # If all machine-involved edges are locked, then machine itself can be 100% locked
        # If not all machine-involved sides are locked, do some complicated logic/guessing/ask user (TODO:)
        numbered_nodes = [i for i, x in self.recipes.items() if getattr(x, 'number', False)]
        need_locking = {i for i in self.recipes.keys() if i not in numbered_nodes and i not in {'sink', 'source'}}

        if len(numbered_nodes) == 0:
            raise RuntimeError('Need at least one "number" argument to base machine balancing around.')

        print(numbered_nodes)
        print(need_locking)
        print()

        # First lock all edges adj to numbered nodes
        for rec_id in numbered_nodes:
            rec = self.recipes[rec_id]
            connected_edges = adj[str(rec_id)]

            # Multiply I/O and eut
            self.recipes[rec_id] *= getattr(rec, 'number') # NOTE: Sets rec.multiplier

            # Color edge as "locked"
            self.nodes[rec_id].update({'fillcolor': 'green'})
            existing_label = self.nodes[rec_id]['label']
            self.nodes[rec_id]['label'] = f'{rec.multiplier}x {rec.user_voltage} {existing_label}\nCycle: {rec.dur/20}s\nEU/t: {rec.eut}'

            # Lock all adjacent ingredient edges
            self._simpleLockMachineEdges(str(rec_id), rec) # Used when multiplier is known
            self.createAdjacencyList()

        while need_locking:
            # Now propagate updates throughout the tree
            # Prefer sides with maximum information (highest ratio of determined edges to total edges)
            # Compute determined edges for all machines
            determined_edge_count = defaultdict(dict)
            for rec_id in need_locking:
                rec = self.recipes[rec_id]
                determined_edge_count[rec_id]['I'] = [
                    sum([1 for edge in self.adj_machine[rec_id]['I'] if self.edges[edge].get('locked', False)]),
                    len(self.adj_machine[rec_id]['I']),
                ]
                determined_edge_count[rec_id]['O'] = [
                    sum([1 for edge in self.adj_machine[rec_id]['O'] if self.edges[edge].get('locked', False)]),
                    len(self.adj_machine[rec_id]['O']),
                ]

            print(determined_edge_count)

            # Now pick in this order:
            # 1. Edges with complete side determination, using total edge determination ratio as tiebreaker
            # 2. Edges with incomplete side determination, but highest total edge determination ratio

            # total_determination_score = sorted(determined_edge_count.items(), reverse=True, key=lambda x: x[1][0] / x[1][1])
            determination_score = {
                rec_id: [
                    # Complete side determination count
                    sum([
                        (
                            determined_edge_count[rec_id][io_type][0] // determined_edge_count[rec_id][io_type][1]
                            if determined_edge_count[rec_id][io_type][1] != 0
                            else 0
                        )
                        for io_type in ['I', 'O']
                    ]),
                    # Total edge determination ratio
                    sum([determined_edge_count[rec_id][io_type][0] for io_type in ['I', 'O']])
                    /
                    sum([determined_edge_count[rec_id][io_type][1] for io_type in ['I', 'O']])
                ]
                for rec_id
                in determined_edge_count
            }
            edge_priority = sorted([
                    [stats, rec_id]
                    for rec_id, stats
                    in determination_score.items()
                ],
                reverse=True,
                key=lambda x: x[0]
            )
            picked_edge = edge_priority[0]
            if picked_edge[0][1] > 0: # At least one determined edge
                rec_id = picked_edge[1]
                rec = self.recipes[rec_id]
                self._lockMachine(rec_id, rec)
                need_locking.remove(rec_id)

                if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
                    self.outputGraphviz()
            else:
                cprint('Unable to compute some of the tree due to missing information; refer to output graph.', 'red')
                break

            self.createAdjacencyList()

        if self.graph_config.get('POWER_LINE', False):
            self._addPowerLineNodes()
        self._addIONode()


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
            # 'creosote oil': 8_000,
            # 'biomass': 8_000,
            # 'oil': 16_000,
            'sulfuric light fuel': 40_000,
            'octane': 80_000,
            'methanol': 84_000,
            'ethanol': 192_000,
            'bio diesel': 256_000,
            'light fuel': 305_000,
            'diesel': 480_000,
            'ether': 537_000,
            'gasoline': 576_000,
            'cetane-boosted diesel': 1_000_000,
            'ethanol gasoline': 750_000,
            'butanol': 1_125_000,
            'jet fuel no.3': 1_324_000,
            'high octane gasoline': 1_728_000,
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
                cprint(f'Detected burnable: {ing_name.title()}! Adding to chart.', 'blue')
                generator_idx, eut_per_cell = known_burnables[ing_name]
                gen_name = generator_names[generator_idx].title()

                # Add node
                node_id = f'{generator_number}-{generator_idx}'
                generator_number += 1
                node_name = f'{gen_name} (100% eff)'
                self.addNode(
                    node_id,
                    label= node_name,
                    fillcolor='lightblue2',
                )

                # Fix edges to point at said node
                produced_eut = eut_per_cell * quant / 1000
                print(quant, eut_per_cell, produced_eut)
                # Edge (old output) -> (generator)
                self.addEdge(
                    node_from,
                    node_id,
                    ing_name,
                    quant,
                    **edge_data['kwargs'],
                )
                # Edge (generator) -> (EU sink)
                self.addEdge(
                    node_id,
                    'sink',
                    'EU',
                    produced_eut,
                )
                # Remove old edge and repopulate adjacency list
                del self.edges[edge]
                self.createAdjacencyList()


    def _addIONode(self):
        # Now that tree is fully locked, add I/O node
        # Specifically, inputs are adj[source] and outputs are adj[sink]

        def makeLineHtml(color, text, amt_text):
            return f'<tr><td align="left"><font color="{color}">{text}</font></td><td align ="right"><font color="{color}">{amt_text}</font></td></tr>'

        self.createAdjacencyList()

        # Compute I/O
        inputs = self.adj['source']
        outputs = self.adj['sink']
        total_io = defaultdict(float)
        for direction in [-1, 1]:
            if direction == -1:
                # Inputs
                edges = self.adj['source']['O']
            elif direction == 1:
                # Outputs
                edges = self.adj['sink']['I']

            for edge in edges:
                from_node, to_node, ing_name = edge
                edge_data = self.edges[edge]
                quant = edge_data['quant']
                total_io[ing_name] += direction * quant

        # Create I/O lines
        io_label_lines = []
        io_label_lines.append('<tr><td align="left"><font color="white">Summary</font></td></tr>')
        for name, quant in sorted(total_io.items(), key=lambda x: x[1]):
            if name == 'EU':
                continue

            # Skip if too small (intended to avoid floating point issues)
            near_zero_range = 10**-7
            if -near_zero_range < quant < near_zero_range:
                continue

            amt_text = f'{self.NDecimals(quant, 2)}/s'
            if quant < 0:
                io_label_lines.append(makeLineHtml('red', name.title(), amt_text))
            else:
                io_label_lines.append(makeLineHtml('MediumSeaGreen', name.title(), amt_text))

        # Compute total EU/t cost and (if power line) output
        total_eut = 0
        for rec in self.recipes.values():
            total_eut += rec.eut
        io_label_lines.append('<HR/>')
        eut_rounded = int(math.ceil(total_eut))
        io_label_lines.append(makeLineHtml('red', 'Input EU/t:', eut_rounded))
        if 'EU' in total_io:
            produced_eut = int(math.floor(total_io['EU'] / 20))
            io_label_lines.append(makeLineHtml('MediumSeaGreen', 'Output EU/t:', produced_eut))
            io_label_lines.append('<HR/>')
            net_eut = produced_eut - eut_rounded
            if net_eut > 0:
                io_label_lines.append(makeLineHtml('MediumSeaGreen', 'Net EU/t:', net_eut))
            else:
                io_label_lines.append(makeLineHtml('red', 'Net EU/t:', net_eut))

        # Create final table
        io_label = ''.join(io_label_lines)
        io_label = f'<<table border="0">{io_label}</table>>'

        # Add to graph
        self.addNode(
            'total_io_node',
            label=io_label,
            fillcolor=self.darkModeColor,
        )


    @staticmethod
    def NDecimals(number, N):
        if abs(number) >= 1:
            return round(number, N)
        else:
            return sigfig_round(number, N)


    def _lockMachine(self, rec_id, rec, determined=False):
        # Compute multipliers based on all locked edges (other I/O stream as well if available)
        all_relevant_edges = {
            'I': [x for x in self.adj_machine[rec_id]['I'] if self.edges[x].get('locked', False)],
            'O': [x for x in self.adj_machine[rec_id]['O'] if self.edges[x].get('locked', False)],
        }
        print(all_relevant_edges)

        if all(len(y) == 0 for x, y in all_relevant_edges.items()):
            cprint(f'No locked machine edges adjacent to {rec.machine.title()}. Cannot balance.', 'red')
            self.outputGraphviz()
            exit(1)

        multipliers = []
        for io_type in ['I', 'O']:
            io_side_edges = all_relevant_edges[io_type]
            total_sided_request = defaultdict(float) # Want to handle multiple ingredient inputs properly

            for edge in io_side_edges:
                node_from, node_to, ing_name = edge
                if io_type == 'I':
                    other_rec = self.recipes[node_from]
                elif io_type == 'O':
                    other_rec = self.recipes[node_to]

                wanted_quant = sum(getattr(other_rec, swapIO(io_type))[ing_name])
                wanted_per_s = wanted_quant / (other_rec.dur / 20)

                total_sided_request[ing_name] += wanted_per_s

            for ing, quant_per_s in total_sided_request.items():
                base_speed = sum(getattr(rec, io_type)[ing]) / (rec.dur / 20)
                multipliers.append(quant_per_s / base_speed)

        print(rec.machine, multipliers)
        final_multiplier = max(multipliers)
        self.recipes[rec_id] *= final_multiplier

        existing_label = self.nodes[rec_id]['label']
        self.nodes[rec_id]['label'] = '\n'.join([
            f'{round(rec.multiplier, 2)}x {rec.user_voltage} {existing_label}',
            f'Cycle: {rec.dur/20}s',
            f'EU/t: {round(rec.eut, 2)}',
        ])

        # Lock ingredient edges using new quant
        self._lockMachineEdges(rec_id, rec)


    def _lockMachineEdges(self, rec_id, rec):
        # Lock all adjacent edges to a particular recipe
        # Do this process per-ingredient - there can be multiple input or output edges for a particular ingredient
        # By the time this function is called, self.adj and self.adj_machine should already exist
        # rec.multiplier should already be determined

        # ==[ ALGORITHM ]==
        # For input edges
            # If single input
                # If undetermined, lock
                # If determined
                    # If determined = ing request quant (ish)
                        # do nothing
                    # If determined > ing request quant
                        # send to sink
                    # If determined < ing request quant
                        # take from source
            # If multiple input
                # If all determined
                    # Follow single input determined rules, except sum(determined)
                # If all but one determined
                    # If sum(determined) < ing request quant
                        # request remainder from undetermined
                    # If sum(determined) >= ing request quant
                        # ((this might indicate an error, throw warning))
                        # the most reasonable response would probably be:
                        # 1. reorganize edges from determined to go to sink (somewhat complicated process)
                            # there may be a fractional edge
                        # 2. send undetermined edge to sink
                # If >1 undetermined
                    # no way to figure out what was meant here
                    # throw error and ask user to specify additional information
        # For output edges
            # If single output
                # If undetermined, lock
                # If determined
                    # If determined > ing supply quant
                        # take from source
                    # If determined < ing supply quant
                        # send excess to sink
            # If multiple output
                # If all determined
                    # Follow single output rules, except sum(determined)
                # If all but one determined
                    # If sum(determined) < ing supply quant
                        # send remainder to undetermined
                    # If sum(determined) >= ing supply quant
                        # (another difficult situation)
                        # (just throw an error for now)
                        # Some sketch work at a solutions:
                            # 1. send some from source to supply missing determined
                            # 2. remove edge to undetermined
                            # 3.

                            # are there other sources of this item/fluid?
                                # if no
                                    # supply undetermined and missing determined from source
                                # if yes, are the other supply nodes locked?
                                    # if all locked
                                        # if sufficient supply from other locked nodes
                                            # remove edge with current rec, other machines will supply
                                        # if insufficient supply from other locked nodes
                                            #
                # If >1 undetermined
                    # no way to determine what was meant here
                    # throw error and ask user to specify additional information


        adj_edges = self.adj[rec_id]
        # Create mapping of {io_dir: {ing_name: edges}}
        ing_edges = {
            'I': defaultdict(list),
            'O': defaultdict(list),
        }
        for io_dir in ['I', 'O']:
            for edge in adj_edges[io_dir]:
                node_from, node_to, ing_name = edge
                ing_edges[io_dir][ing_name].append(edge)

        for io_dir in ['I', 'O']:
            for ing_name, edges in ing_edges[io_dir].items():
                num_io = len(edges)
                locked_bools = [self.edges[x].get('locked', False) for x in edges]
                machine_ing_io = sum(getattr(rec, io_dir)[ing_name]) / (rec.dur / 20)

                if io_dir == 'I':
                    if num_io == 1: # Single input
                        if locked_bools[0] == False: # Undetermined
                            self.edges[edges[0]]['quant'] = machine_ing_io
                            self.edges[edges[0]]['locked'] = True
                        else: # Determined
                            locked_quant = self.edges[edges[0]]['quant']
                            excess = locked_quant - machine_ing_io
                            node_from, node_to, _ = edges[0]

                            if math.isclose(excess, 0, abs_tol=1e-9):
                                continue
                            elif excess > 0:
                                # 1. Adjust locked edge down to actual io
                                self.edges[edges[0]]['quant'] -= excess
                                # 2. Send remainder to sink
                                self.addEdge(
                                    node_from,
                                    'sink',
                                    ing_name,
                                    excess
                                )
                            elif excess < 0:
                                # Get missing amount from source
                                self.addEdge(
                                    'source',
                                    node_to,
                                    ing_name,
                                    -excess
                                )
                    else: # Multiple input
                        if all(locked_bools): # All inputs determined
                            edge_quants = [self.edges[x]['quant'] for x in edges]
                            locked_quant = sum(edge_quants)
                            excess = locked_quant - machine_ing_io # Excess ingredient available

                            if math.isclose(excess, 0, abs_tol=1e-9):
                                continue
                            elif excess < 0:
                                # Get missing amount from source
                                self.addEdge(
                                    'source',
                                    node_to,
                                    ing_name,
                                    -excess
                                )
                            elif excess > 0:
                                # Adjust connected edges down until excess is satisfied
                                # If math doesn't work out without remainder, adjust relevant edge down
                                    # and make a new sink

                                for idx, quant in edge_quants:
                                    relevant_edge = edges[idx]
                                    node_from, node_to, _ = relevant_edge
                                    excess -= quant
                                    if excess > 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                        # Send entire edge to sink and then continue iteration
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant
                                        )
                                        del self.edges[relevant_edge]
                                        if math.isclose(excess, 0, abs_tol=1e-9):
                                            break
                                    else: # Removing edge would cause negative excess, need to make fractional edge
                                        excess *= -1
                                        self.edges[relevant_edge]['quant'] -= excess
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant - excess
                                        )
                        elif sum(locked_bools) == len(edges) - 1: # 1 input undetermined
                            edge_quants = {x: self.edges[x]['quant'] for x in edges if self.edges[x].get('locked', False)}
                            locked_quant = sum(edge_quants.values())
                            excess = locked_quant - machine_ing_io # Excess ingredient available
                            unlocked_edge = edges[locked_bools.index(False)]

                            if excess > 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                # Get rid of link to undetermined edge and then perform same process as all determined
                                del self.edges[unlocked_edge]

                                if math.isclose(excess, 0, abs_tol=1e-9):
                                    continue

                                for edge, quant in edge_quants.items():
                                    node_from, node_to, _ = edge
                                    excess -= quant
                                    if excess > 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                        # Send entire edge to sink and then continue iteration
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant,
                                            locked=True
                                        )
                                        del self.edges[edge]
                                        if math.isclose(excess, 0, abs_tol=1e-9):
                                            break
                                    else: # Removing edge would cause negative excess, need to make fractional edge
                                        excess *= -1
                                        self.edges[edge]['quant'] -= excess
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant - excess,
                                            locked=True
                                        )

                            elif excess < 0: # Not enough product from locked edges, therefore must come from unlocked
                                self.edges[unlocked_edge]['quant'] = -excess
                                self.edges[unlocked_edge]['locked'] = True
                        else:
                            cprint('Too many undetermined edges! Please define more numbered nodes (or different ones).', 'red')
                            cprint(f'Problem: {len(edges) - sum(locked_bools)} edges are undetermined. Can only handle 1 at most.', 'red')
                            cprint(f'Inputs for: {rec}', 'red')
                            cprint(f'Input edges: {edges}', 'red')

                            self.createAdjacencyList()
                            self.outputGraphviz()
                            exit(1)
                elif io_dir == 'O':
                    if num_io == 1: # Single input
                        if locked_bools[0] == False: # Undetermined
                            self.edges[edges[0]]['quant'] = machine_ing_io
                            self.edges[edges[0]]['locked'] = True
                        else: # Determined
                            locked_quant = self.edges[edges[0]]['quant']
                            excess = machine_ing_io - locked_quant # Excess rec ingredient available
                            node_from, node_to, _ = edges[0]

                            if math.isclose(excess, 0, abs_tol=1e-9):
                                continue
                            elif excess > 0:
                                # Send remainder to sink
                                self.addEdge(
                                    node_from,
                                    'sink',
                                    ing_name,
                                    excess,
                                    locked=True
                                )
                            elif excess < 0:
                                # Get missing amount from source
                                self.addEdge(
                                    'source',
                                    node_to,
                                    ing_name,
                                    -excess,
                                    locked=True
                                )
                    else:
                        if all(locked_bools): # All inputs determined
                            edge_quants = [self.edges[x]['quant'] for x in edges]
                            locked_quant = sum(edge_quants)
                            excess = machine_ing_io - locked_quant # Excess rec ingredient available

                            if math.isclose(excess, 0, abs_tol=1e-9):
                                continue
                            elif excess < 0:
                                # Fill as many edges as possible, fill rest from source
                                # FIXME: This is still doing single logic
                                self.addEdge(
                                    'source',
                                    node_to,
                                    ing_name,
                                    -excess
                                )
                            elif excess > 0:
                                # Adjust connected edges down until excess is satisfied
                                # If math doesn't work out without remainder, adjust relevant edge down
                                    # and make a new sink

                                for idx, quant in edge_quants:
                                    relevant_edge = edges[idx]
                                    node_from, node_to, _ = relevant_edge
                                    excess -= quant
                                    if excess > 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                        # Send entire edge to sink and then continue iteration
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant,
                                            locked=True,
                                        )
                                        del self.edges[relevant_edge]
                                        if math.isclose(excess, 0, abs_tol=1e-9):
                                            break
                                    else: # Removing edge would cause negative excess, need to make fractional edge
                                        self.edges[relevant_edge]['quant'] -= excess
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            excess,
                                            locked=True,
                                        )
                        elif sum(locked_bools) == len(edges) - 1: # 1 input undetermined
                            edge_quants = {x: self.edges[x]['quant'] for x in edges if self.edges[x].get('locked', False)}
                            locked_quant = sum(edge_quants.values())
                            excess = machine_ing_io - locked_quant # Excess rec ingredient available
                            unlocked_edge = edges[locked_bools.index(False)]

                            if excess < 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                # Get rid of link to undetermined edge and then perform same process as all determined
                                del self.edges[unlocked_edge]

                                if math.isclose(excess, 0, abs_tol=1e-9):
                                    continue

                                for edge, quant in edge_quants.items():
                                    node_from, node_to, _ = edge
                                    excess += quant
                                    if excess < 0 or math.isclose(excess, 0, abs_tol=1e-9):
                                        # Get entire edge from source and continue iteration
                                        self.addEdge(
                                            'source',
                                            node_to,
                                            ing_name,
                                            quant,
                                            locked=True,
                                        )
                                        del self.edges[edge]
                                        if math.isclose(excess, 0, abs_tol=1e-9):
                                            break
                                    else: # Removing edge would cause too much excess, need to make fractional edge
                                        self.edges[edge]['quant'] -= excess
                                        self.addEdge(
                                            node_from,
                                            'sink',
                                            ing_name,
                                            quant - excess,
                                            locked=True,
                                        )

                            elif excess > 0: # Send excess to unlocked node
                                self.edges[unlocked_edge]['quant'] = excess
                                self.edges[unlocked_edge]['locked'] = True
                        else:
                            cprint('Too many undetermined edges! Please define more numbered nodes (or different ones).', 'red')
                            cprint(f'Problem: {len(edges) - sum(locked_bools)} edges are undetermined. Can only handle 1 at most.', 'red')
                            cprint(f'Outputs for: {rec}', 'red')
                            cprint(f'Output edges: {edges}', 'red')

                            self.createAdjacencyList()
                            self.outputGraphviz()
                            exit(1)


    def _simpleLockMachineEdges(self, rec_id, rec):
        # _lockMachineEdges, but no information requirements - just force lock the edges
        for io_dir in ['I', 'O']:
            for edge in self.adj[rec_id][io_dir]:
                node_from, node_to, ing_name = edge
                edge_locked = self.edges[edge].get('locked', False)

                packet_quant = sum(getattr(rec, io_dir)[ing_name]) / (rec.dur / 20)
                if not edge_locked:
                    self.edges[edge]['quant'] = packet_quant
                else:
                    # Edge is already locked, which means:
                    # If packet sent from destination ("request")
                        # if packet > locked then get from source
                        # if packet < locked then send to sink
                    # If packet sent from src ("supply")
                        # if packet > locked then send to sink
                        # if packet < locked then get from source

                    if math.isclose(packet_quant, self.edges[edge]['quant']):
                        self.edges[edge]['quant'] = packet_quant
                        self.edges[edge]['locked'] = True
                        continue

                    locked_quant = self.edges[edge]['quant']
                    packet_diff = abs(packet_quant - locked_quant)
                    if io_dir == 'I':
                        if packet_quant > locked_quant:
                            self.addEdge(
                                'source',
                                node_to,
                                ing_name,
                                packet_diff,
                                locked=True,
                            )
                        else:
                            self.addEdge(
                                node_from,
                                'sink',
                                ing_name,
                                packet_diff,
                                locked=True,
                            )
                    if io_dir == 'O':
                        if packet_quant > locked_quant:
                            self.addEdge(
                                node_from,
                                'sink',
                                ing_name,
                                packet_diff,
                                locked=True,
                            )
                        else:
                            self.addEdge(
                                'source',
                                node_to,
                                ing_name,
                                packet_diff,
                                locked=True,
                            )

                self.edges[edge]['locked'] = True

        if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
            self.outputGraphviz()


    def outputGraphviz(self):
        # Outputs a graphviz png using the graph info
        node_style = {
            'shape': 'box',
            'style': 'filled',
        }
        g = graphviz.Digraph(
            engine='dot',
            strict=False, # Prevents edge grouping
            graph_attr={
                # 'splines': 'ortho',
                # 'rankdir': 'TD',
                # 'ranksep': '0.5',
                # 'overlap': 'scale',
                'bgcolor': self.darkModeColor,
                # 'mindist': '0.1',
                # 'overlap': 'false',
            }
        )

        # Populate nodes
        for rec_id, kwargs in self.nodes.items():
            if isinstance(rec_id, int):
                g.node(
                    str(rec_id),
                    **kwargs,
                    **node_style
                )
            elif isinstance(rec_id, str):
                g.node(
                    str(rec_id),
                    **kwargs,
                    **node_style
                )

        # Populate edges
        edgecolor_cycle = [
            'white',
            'orange',
            'yellow',
            'green',
            'violet',
        ]
        if self.graph_config.get('USE_RAINBOW_EDGES', None):
            cycle_obj = itertools.cycle(edgecolor_cycle)
        else:
            cycle_obj = itertools.cycle(['white'])
        ingredient_colors = {}

        capitalization_exceptions = {
            'eu': 'EU',
        }
        unit_exceptions = {
            'eu': lambda eu: f'{int(math.floor(eu / 20))}/t'
        }
        for io_info, edge_data in self.edges.items():
            node_from, node_to, ing_name = io_info
            ing_quant, kwargs = edge_data['quant'], edge_data['kwargs']

            ing_name = ing_name.lower()

            # Make quantity label
            if ing_name in unit_exceptions:
                quant_label = unit_exceptions[ing_name](ing_quant)
            else:
                quant_label = f'{self.NDecimals(ing_quant, 2)}/s'

            # Make ingredient label
            if ing_name in capitalization_exceptions:
                ing_name = capitalization_exceptions[ing_name.lower()]
            else:
                ing_name = ing_name.title()

            # Strip bad arguments
            if 'locked' in kwargs:
                del kwargs['locked']

            # Assign ing color if it doesn't already exist
            if ing_name not in ingredient_colors:
                ingredient_colors[ing_name] = next(cycle_obj)
            g.edge(
                node_from,
                node_to,
                label=f'{ing_name}\n({quant_label})',
                fontcolor=ingredient_colors[ing_name],
                color=ingredient_colors[ing_name],
                **kwargs,
            )

        # Output final graph
        g.render(
            self.graph_name,
            'output/',
            view=True,
            format=self.graph_config['OUTPUT_FORMAT'],
        )

        if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
            input()
