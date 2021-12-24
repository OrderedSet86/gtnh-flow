# Graphs are traditionally collections of nodes and edges.
# However, recipes represent a node and all its connected edges.
# As a result, a special algorithm is required to connect all relevant edges in a factory.

# Standard libraries
from collections import defaultdict

# Pypi libraries
import graphviz

# Internal libraries
from dataClasses.base import Ingredient, IngredientCollection, Recipe


def swapIO(io_type):
    if io_type == 'I':
        return 'O'
    elif io_type == 'O':
        return 'I'
    else:
        raise RuntimeError(f'Improper I/O type: {io_type}')

class Graph:
    # Assumes recipes are pre-overclocked appropriately

    def __init__(self, graph_name, recipes):
        self.graph_name = graph_name
        self.recipes = recipes
        self.total_IO = IngredientCollection()
        self.nodes = []
        self.edges = {} # uniquely defined by (machine from, machine to, ing name)


    def addNode(self, node_id, **kwargs):
        self.nodes.append([node_id, kwargs])
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
        self.addNode('source', fillcolor='ghostwhite')
        self.addNode('sink', fillcolor='ghostwhite')

        # Compute {[ingredient name][IO direction] -> involved recipes} table
        involved_recipes = defaultdict(lambda: defaultdict(list))
        for rec_id, rec in enumerate(self.recipes):
            for io_type in ['I', 'O']:
                for ing in getattr(rec, io_type):
                    involved_recipes[ing.name][io_type].append(rec_id)

        # Add I/O connections
        for rec_id, rec in enumerate(self.recipes):
            self.addNode(
                str(rec_id),
                fillcolor='lightblue2',
                label=rec.machine,
            )
            for io_type in ['I', 'O']:
                for ing in getattr(rec, io_type):
                    linked_machines = involved_recipes[swapIO(io_type)]
                    if len(linked_machines) == 0:
                        if io_type == 'I':
                            linked_machines = ['source']
                        elif io_type == 'O':
                            linked_machines = ['sink']

                    for link_id in linked_machines:
                        if io_type == 'I':
                            self.addEdge(
                                str(link_id),
                                str(rec_id),
                                ing.name,
                                ing.quant,
                            )
                        elif io_type == 'O':
                            self.addEdge(
                                str(rec_id),
                                str(link_id),
                                ing.name,
                                ing.quant,
                            )


    def balanceGraph(self):
        # Applies locking info to existing graph
        pass


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

        print(self.nodes)
        print(self.edges)

        for node_id, kwargs in self.nodes:
            g.node(
                node_id,
                **kwargs,
                **node_style
            )
        for io_info, edge_data in self.edges.items():
            node_from, node_to, ing_name = io_info
            ing_quant, kwargs = edge_data['quant'], edge_data['kwargs']
            g.edge(node_from, node_to, **kwargs)

        # Output final graph
        g.render(
            self.graph_name,
            'output/',
            view=True,
            format='png',
        )
