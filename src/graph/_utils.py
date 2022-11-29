import re
from collections import defaultdict

from src.data.basicTypes import Recipe


def swapIO(io_type):
    if io_type == 'I':
        return 'O'
    elif io_type == 'O':
        return 'I'
    else:
        raise RuntimeError(f'Improper I/O type: {io_type}')


def addNode(self, recipe_id, **kwargs):
    self.nodes[recipe_id] = kwargs


def addEdge(self, node_from, node_to, ing_name, quantity, **kwargs):
    self.edges[(node_from, node_to, ing_name)] = {
        'quant': quantity,
        'kwargs': kwargs
    }


def userRound(number):
    # Display numbers nicely for end user (eg. 814.3k)
    # input int/float, return string
    cutoffs = {
        1_000_000_000: lambda x: f'{round(x/1_000_000_000, 2)}B',
        1_000_000: lambda x: f'{round(x/1_000_000, 2)}M',
        1_000: lambda x: f'{round(x/1_000, 2)}K',
        0: lambda x: f'{round(x, 2)}'
    }

    for n, roundfxn in cutoffs.items():
        if abs(number) >= n:
            rounded = roundfxn(number)
            return rounded

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

    self.parent_context.cLog('Recomputing adjacency list...', 'blue')
    for machine, io_group in self.adj_machine.items():
        machine_name = ''
        recipe_obj = self.recipes.get(machine)
        if isinstance(recipe_obj, Recipe):
            machine_name = recipe_obj.machine

        self.parent_context.cLog(f'{machine} {machine_name}', 'blue')
        for io_type, edges in io_group.items():
            self.parent_context.cLog(f'{io_type} {edges}', 'blue')
    self.parent_context.cLog('')


def _checkIfMachine(self, rec_id):
    # TODO: Memoize calls
    if rec_id in {'source', 'sink', 'total_io_node'}:
        return False
    elif rec_id.startswith(('power_', 'joint_')):
        return False
    return True


def _iterateOverMachines(self):
    # Iterate over non-source/sink noedes and non power nodes
    for rec_id in self.nodes:
        if self._checkIfMachine(rec_id):
            yield self.recipes[rec_id]

