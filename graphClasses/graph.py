# Graphs are traditionally collections of nodes and edges.
# However, recipes represent a node and all its connected edges.
# As a result, a special algorithm is required to connect all relevant edges in a factory.

# Standard libraries
import math
from collections import defaultdict, OrderedDict

# Pypi libraries
import graphviz
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
        self.recipes = recipes
        self.total_IO = IngredientCollection() # TODO:
        self.nodes = {}
        self.edges = {} # uniquely defined by (machine from, machine to, ing name)
        self.graph_config = graph_config
        if self.graph_config == None:
            self.graph_config = {}

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
        for rec_id, rec in enumerate(self.recipes):
            for io_type in ['I', 'O']:
                for ing in getattr(rec, io_type):
                    involved_recipes[ing.name][io_type].append(rec_id)

        # Add I/O connections
        added_edges = set()
        for rec_id, rec in enumerate(self.recipes):
            self.addNode(
                rec_id,
                fillcolor='lightblue2',
                label=f'({rec_id}) {rec.machine.title()}',
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


    def balanceGraph(self):
        # Applies locking info to existing graph
        self.removeBackEdges()

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

        # Debug
        for node, adj_edges in adj_machine.items():
            if node in ['sink', 'source']:
                continue
            print(node, dict(adj_edges))
        print()

        # Locking rules:
        # If all machine-involved edges are locked, then machine itself can be 100% locked
        # If not all machine-involved sides are locked, do some complicated logic/guessing/ask user (TODO:)
        numbered_nodes = [i for i, x in enumerate(self.recipes) if getattr(x, 'number', False)]
        need_locking = {str(i) for i in range(len(self.recipes)) if i not in numbered_nodes and i not in {'sink', 'source'}}

        if len(numbered_nodes) == 0:
            raise RuntimeError('Need at least one "number" argument to base machine balancing around.')

        print(numbered_nodes)
        print(need_locking)
        print()

        # First lock all edges adj to numbered nodes
        for rec_id in numbered_nodes:
            rec = self.recipes[rec_id]
            connected_edges = adj[str(rec_id)]
            # print(rec_id, connected_edges)

            # Multiply I/O and eut
            self.recipes[rec_id] *= getattr(rec, 'number')
            # print(self.recipes[rec_id])
            # print(rec)

            # Color edge as "locked"
            self.nodes[rec_id].update({'fillcolor': 'green'})
            existing_label = self.nodes[rec_id]['label']
            self.nodes[rec_id]['label'] = f'{rec.multiplier}x {rec.user_voltage} {existing_label}\nCycle: {rec.dur/20}s\nEU/t: {rec.eut}'

            # Lock all adj edges
            for io_dir in connected_edges:
                for edge in connected_edges[io_dir]:
                    self._lockEdgeQuant(edge, rec, io_dir)

        # Now propagate updates throughout the tree
        while need_locking:
            found_lockable = False

            for rec_id in need_locking:
                rec = self.recipes[int(rec_id)]
                for io_type in ['I', 'O']:
                    relevant_machine_edges = adj_machine[rec_id][io_type]
                    if len(relevant_machine_edges) > 0 and all([self.edges[x].get('locked', False) for x in relevant_machine_edges]):
                        # This machine is available to be locked!
                        found_lockable = True

                        # Compute multipliers based on all locked edges (other I/O stream as well if available)
                        all_relevant_edges = {
                            'I': [x for x in adj_machine[rec_id]['I'] if self.edges[x].get('locked', False)],
                            'O': [x for x in adj_machine[rec_id]['O'] if self.edges[x].get('locked', False)],
                        }
                        print(all_relevant_edges)

                        if len(all_relevant_edges) == 0:
                            cprint(f'No machines adjacent to {rec.machine}. Cannot balance.', 'red')
                            return

                        multipliers = []
                        for io_type in ['I', 'O']:
                            io_side_edges = all_relevant_edges[io_type]
                            for edge in io_side_edges:
                                node_from, node_to, ing_name = edge
                                if io_type == 'I':
                                    other_rec = self.recipes[int(node_from)]
                                elif io_type == 'O':
                                    other_rec = self.recipes[int(node_to)]

                                wanted_quant = getattr(other_rec, swapIO(io_type))[ing_name]
                                wanted_per_s = wanted_quant / (other_rec.dur / 20)

                                base_speed = getattr(rec, io_type)[ing_name] / (rec.dur / 20)
                                multipliers.append(wanted_per_s / base_speed)

                        print(rec.machine, multipliers)
                        final_multiplier = max(multipliers)
                        self.recipes[int(rec_id)] *= final_multiplier

                        existing_label = self.nodes[int(rec_id)]['label']
                        self.nodes[int(rec_id)]['label'] = f'{round(rec.multiplier, 2)}x {rec.user_voltage} {existing_label}\nCycle: {rec.dur/20}s\nEU/t: {round(rec.eut, 2)}'

                        # Lock edges using new quant
                        connected_edges = adj[rec_id]
                        for io_dir in connected_edges:
                            for edge in connected_edges[io_dir]:
                                self._lockEdgeQuant(edge, rec, io_dir)

                        # Pop recipe and restart iteration
                        need_locking.remove(rec_id)
                        break

                    if found_lockable:
                        break

                if found_lockable:
                    break

            if found_lockable:
                continue

            cprint('Unable to compute some of the tree due to missing information; refer to output graph.', 'red')
            break




    def _lockEdgeQuant(self, edge, rec, io_dir):
        node_from, node_to, ing_name = edge
        edge_locked = self.edges[edge].get('locked', False)

        packet_quant = getattr(rec, io_dir)[ing_name] / (rec.dur / 20)
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
                return

            locked_quant = self.edges[edge]['quant']
            packet_diff = abs(packet_quant - locked_quant)
            if io_dir == 'I':
                if packet_quant > locked_quant:
                    self.addEdge(
                        'source',
                        node_to,
                        ing_name,
                        packet_diff,
                    )
                else:
                    self.addEdge(
                        node_from,
                        'sink',
                        ing_name,
                        packet_diff,
                    )
            if io_dir == 'O':
                if packet_quant > locked_quant:
                    self.addEdge(
                        node_from,
                        'sink',
                        ing_name,
                        packet_diff,
                    )
                else:
                    self.addEdge(
                        'source',
                        node_to,
                        ing_name,
                        packet_diff,
                    )

        self.edges[edge]['locked'] = True


    def outputGraphviz(self):
        # Outputs a graphviz png using the graph info
        node_style = {
            'shape': 'box',
            'style': 'filled',
        }
        g = graphviz.Digraph(
            strict=False, # Prevents edge grouping
            graph_attr={
            #     'splines': 'ortho'
                'rankdir': 'TD',
                'ranksep': '0.5',
                # 'overlap': 'scale',
            }
        )

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
        for io_info, edge_data in self.edges.items():
            node_from, node_to, ing_name = io_info
            ing_quant, kwargs = edge_data['quant'], edge_data['kwargs']
            g.edge(
                node_from,
                node_to,
                label=f'{ing_name.title()}\n({round(ing_quant, 2)}/s)',
                **kwargs
            )

        # Output final graph
        g.render(
            self.graph_name,
            'output/',
            view=True,
            format='png',
        )
