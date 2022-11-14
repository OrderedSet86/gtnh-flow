import logging
import math
from collections import defaultdict

from termcolor import cprint

from src.graph._utils import swapIO


def _lockMachine(self, rec_id, rec, determined=False):
    # Compute multipliers based on all locked edges (other I/O stream as well if available)
    all_relevant_edges = {
        'I': [x for x in self.adj_machine[rec_id]['I'] if self.edges[x].get('locked', False)],
        'O': [x for x in self.adj_machine[rec_id]['O'] if self.edges[x].get('locked', False)],
    }
    self.parent_context.cLog(f'Locking {rec.machine}...', 'green')
    self.parent_context.cLog(all_relevant_edges, 'yellow')

    if all(len(y) == 0 for x, y in all_relevant_edges.items()):
        self.parent_context.cLog(f'No locked machine edges adjacent to {rec.machine.title()}. Cannot balance.', 'red', level=logging.WARNING)
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

    if len(multipliers) == 1:
        self.parent_context.cLog(f'{rec.machine} {multipliers}', 'white', level=logging.DEBUG)
    else:
        self.parent_context.cLog(f'{rec.machine} {multipliers}', 'red', level=logging.WARNING)
    final_multiplier = max(multipliers)
    self.recipes[rec_id] *= final_multiplier

    existing_label = self.nodes[rec_id]['label']
    self.nodes[rec_id]['label'] = '\n'.join([
        f'{round(rec.multiplier, 2)}x {rec.user_voltage} {existing_label}',
        f'Cycle: {rec.dur/20}s',
        f'Amoritized: {self.userRound(int(round(rec.eut, 0)))} EU/t',
        f'Per Machine: {self.userRound(int(round(rec.base_eut, 0)))} EU/t',
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

                            for idx, quant in enumerate(edge_quants):
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

                            for idx, quant in enumerate(edge_quants):
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