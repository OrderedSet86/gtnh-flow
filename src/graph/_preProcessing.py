from collections import defaultdict

from termcolor import colored

from src.graph._backEdges import BasicGraph, dfs
from src.graph._utils import swapIO


def connectGraph(self):
    '''
    Connects recipes without locking the quantities
    '''

    # Create source and sink nodes
    self.addNode('source', fillcolor=self.graph_config['SOURCESINK_COLOR'], label='source')
    self.addNode('sink', fillcolor=self.graph_config['SOURCESINK_COLOR'], label='sink')

    # Compute {[ingredient name][IO direction] -> involved recipes} table
    involved_recipes = defaultdict(lambda: defaultdict(list))
    for rec_id, rec in self.recipes.items():
        for io_type in ['I', 'O']:
            for ing in getattr(rec, io_type):
                involved_recipes[ing.name][io_type].append(rec_id)

    # Create initial nodes
    for rec_id, rec in self.recipes.items():
        # Create machine label
        if self.graph_config['SHOW_MACHINE_INDICES']:
            machine_label = [f'({rec_id}) {rec.machine.title()}']
        else:
            machine_label = [rec.machine.title()]

        # Add lines for the special arguments
        # TODO: Split into its own fxn
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
                machine_label.append(line_generator(rec))

        machine_label = '\n'.join(machine_label)
        machine_color = self.graph_config['MACHINE_COLORS'].get(
            rec.machine, self.graph_config['DEFAULT_MACHINE_COLOR'])
        self.addNode(
            rec_id,
            fillcolor=machine_color,
            label=machine_label
        )

    # Add I/O connections
    added_edges = set()
    for rec_id, rec in self.recipes.items():
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
            self.parent_context.log.info(colored(f'Fixing factory cycle by redirecting "{ing_name.title()}" to sink', 'yellow'))

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
