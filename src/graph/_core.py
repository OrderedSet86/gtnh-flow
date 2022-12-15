import logging
import re
from io import StringIO
from collections import defaultdict

import graphviz



def balanceGraph(self):
    # Applies locking info to existing graph
    self.removeBackEdges()

    # Create adjacency list for easier compute
    self.createAdjacencyList()
    adj = self.adj
    adj_machine = self.adj_machine

    # Debug
    self.parent_context.cLog('All machine-machine edges:')
    for node, adj_edges in adj_machine.items():
        if node in ['sink', 'source']:
            continue
        self.parent_context.cLog(f'{node} {dict(adj_edges)}')
    self.parent_context.cLog('')

    # Locking rules:
    # If all machine-involved edges are locked, then machine itself can be 100% locked
    # If not all machine-involved sides are locked, do some complicated logic/guessing/ask user (TODO:)
    targeted_nodes = [i for i, x in self.recipes.items() if getattr(x, 'target', False) != False]
    numbered_nodes = [i for i, x in self.recipes.items() if getattr(x, 'number', False) != False]
    need_locking = {i for i in self.recipes.keys() if i not in numbered_nodes and i not in targeted_nodes and i not in {'sink', 'source'}}

    ln = len(numbered_nodes)
    lt = len(targeted_nodes)

    if lt == 0 and ln == 0:
        raise RuntimeError('Need at least one "number" or "target" argument to base machine balancing around.')
    elif ln != 0 and lt == 0:
        # First lock all edges adj to numbered nodes
        for rec_id in numbered_nodes:
            rec = self.recipes[rec_id]

            # Multiply I/O and eut
            self.recipes[rec_id] *= getattr(rec, 'number') # NOTE: Sets rec.multiplier

            # Color edge as "locked"
            self.nodes[rec_id].update({'fillcolor': self.graph_config['LOCKEDNODE_COLOR']})
            existing_label = self.nodes[rec_id]['label']
            self.nodes[rec_id]['label'] = '\n'.join([
                f'{round(rec.multiplier, 2)}x {rec.user_voltage} {existing_label}',
                f'Cycle: {rec.dur/20}s',
                f'Amoritized: {self.userRound(int(round(rec.eut, 0)))} EU/t',
                f'Per Machine: {self.userRound(int(round(rec.base_eut, 0)))} EU/t',
            ])

            # Lock all adjacent ingredient edges
            self._simpleLockMachineEdges(str(rec_id), rec) # Used when multiplier is known
            self.createAdjacencyList()

    elif ln == 0 and lt != 0:
        if lt > 1:
            raise NotImplementedError('>1 targeted node not supported')

        rec_id = targeted_nodes[0]
        rec = self.recipes[rec_id]
        target_ings = rec.target
        if len(target_ings) > 1:
            raise NotImplementedError('>1 targeted quantity not supported')

        # Compute machine multiplier based on requested quant
        # (note already OC'd by this time)
        ing, target_quant = list(target_ings.items())[0]
        if ing not in [x.name for x in rec.O._ings]:
            raise RuntimeError(f'Targetted quantity must be in machine outputs for \n{rec}')
        quant_per_tick_at_1x = rec.O[ing][0] / rec.dur
        machine_multiplier = round(target_quant / (quant_per_tick_at_1x * 20), 8)

        # Update on graph
        # Multiply I/O and eut
        self.recipes[rec_id] *= machine_multiplier # NOTE: Sets rec.multiplier

        # Color edge as locked
        self.nodes[rec_id].update({'fillcolor': self.graph_config['LOCKEDNODE_COLOR']})
        existing_label = self.nodes[rec_id]['label']
        self.nodes[rec_id]['label'] = '\n'.join([
            f'{round(rec.multiplier, 2)}x {rec.user_voltage} {existing_label}',
            f'Cycle: {rec.dur/20}s',
            f'Amoritized: {self.userRound(int(round(rec.eut, 0)))} EU/t',
            f'Per Machine: {self.userRound(int(round(rec.base_eut, 0)))} EU/t',
        ])


        # Lock all adjacent ingredient edges
        self._simpleLockMachineEdges(str(rec_id), rec) # Used when multiplier is known
        self.createAdjacencyList()

        # self.outputGraphviz()

    elif ln != 0 and lt != 0:
        raise NotImplementedError('mixed targeted nodes and numbered nodes not supported')

    self.parent_context.cLog(f'Still need locking: {need_locking}', 'red')
    self.parent_context.cLog('')

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

        self.parent_context.cLog(f'Edge determination data:\n{determined_edge_count}', 'green')

        # Now pick in this order:
        # 1. Edges with complete side determination, using total edge determination ratio as tiebreaker
        # 2. Edges with incomplete side determination, but highest total edge determination ratio

        # total_determination_score = sorted(determined_edge_count.items(), reverse=True, key=lambda x: x[1][0] / x[1][1])
        try:
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
        except ZeroDivisionError as e:
            self.outputGraphviz()
            raise RuntimeError('A machine is disconnected from all the others. Please check for typos. A graph will be output.')

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
            self.parent_context.cLog('Unable to compute some of the tree due to missing information; refer to output graph.', 'red', level=logging.WARNING)
            break

        self.createAdjacencyList()

    if self.graph_config.get('POWER_LINE', False):
        self._addPowerLineNodes()
    self._addSummaryNode()

    if self.graph_config.get('COMBINE_INPUTS', False):
        self._combineInputs()
    if self.graph_config.get('COMBINE_OUTPUTS', False):
        self._combineOutputs()


def outputGraphviz(self):
    # Outputs a graphviz png using the graph info
    node_style = {
        'style': 'filled',
        'fontname': self.graph_config['GENERAL_FONT'],
        'fontsize': str(self.graph_config['NODE_FONTSIZE']),
    }
    edge_style = {
        'fontname': self.graph_config['GENERAL_FONT'],
        'fontsize': str(self.graph_config['EDGE_FONTSIZE']),
        'dir': 'both',
        'arrowtail': 'none',
        'arrowhead': 'none',
        'penwidth': '1',
    }
    g = graphviz.Digraph(
        engine='dot',
        strict=False, # Prevents edge grouping
        graph_attr={
            'bgcolor': self.graph_config['BACKGROUND_COLOR'],
            'splines': self.graph_config['LINE_STYLE'],
            'rankdir': self.graph_config['ORIENTATION'],
            'ranksep': self.graph_config['RANKSEP'],
            'nodesep': self.graph_config['NODESEP'],
        }
    )

    # Collect nodes by subgraph grouping
    groups = defaultdict(list)
    groups['no-group'] = []
    for rec_id, kwargs in self.nodes.items():
        repackaged = (rec_id, kwargs)
        if rec_id in self.recipes:
            rec = self.recipes[rec_id]
            if hasattr(rec, 'group'):
                groups[rec.group].append(repackaged)
            else:
                groups['no-group'].append(repackaged)
        else:
            groups['no-group'].append(repackaged)

    def make_table(lab, inputs, outputs):
        is_inverted = self.graph_config['ORIENTATION'] in ['BT', 'RL']
        is_vertical = self.graph_config['ORIENTATION'] in ['TB', 'BT']
        num_inputs = len(inputs) if inputs is not None else 0
        num_outputs = len(outputs) if outputs is not None else 0
        has_input = num_inputs > 0
        has_output = num_outputs > 0
        
        if not has_input and not has_output:
            return (False, lab)

        machine_cell = ['<br />'.join(lab.split('\n'))]
        lines = [
            ('i',inputs), 
            (None,machine_cell), 
            ('o',outputs)
        ]
        if is_inverted:
            lines.reverse()
        lines = [(x,y) for x,y in lines if y]

        
        io = StringIO()
        if is_vertical:
            # Each Row is a table
            io.write('<<table border="0" cellspacing="0">')
            for port_type,line in lines:
                io.write('<tr>')
                io.write('<td>')
                io.write('<table border="0" cellspacing="0">')
                io.write('<tr>')
                for cell in line:
                    if port_type:
                        port_id = self.getPortId(cell, port_type)
                        ing_name = self.getIngLabel(cell)
                        io.write(f'<td border="1" PORT="{port_id}">{self.stripBrackets(ing_name)}</td>')
                    else:
                        io.write(f'<td border="0">{cell}</td>')
                io.write('</tr>')
                io.write('</table>')
                io.write('</td>')
                io.write('</tr>')
            io.write('</table>>')
        else:
            # Each columns is a table
            io.write('<<table border="0" cellspacing="0">')
            io.write('<tr>')
            for port_type,line in lines:
                io.write('<td>')
                io.write('<table border="0" cellspacing="0">')
                for cell in line:
                    io.write('<tr>')
                    if port_type:
                        port_id = self.getPortId(cell, port_type)
                        ing_name = self.getIngLabel(cell)
                        io.write(f'<td border="1" PORT="{port_id}">{self.stripBrackets(ing_name)}</td>')
                    else:
                        io.write(f'<td border="0">{cell}</td>')
                    io.write('</tr>')
                io.write('</table>')
                io.write('</td>')
            io.write('</tr>')
            io.write('</table>>')
        return (True, io.getvalue())

    def add_node_internal(graph, node_name, **kwargs):
        label = kwargs['label'] if 'label' in kwargs else None
        isTable = False
        newLabel = None

        def unique(sequence):
            seen = set()
            return [x for x in sequence if not (x in seen or seen.add(x))]
        
        if node_name == 'source':
            names = unique([name for src,_,name in self.edges.keys() if src == 'source'])
            isTable, newLabel = make_table(label, None, names)
        elif node_name == 'sink':
            names = unique([name for _,dst,name in self.edges.keys() if dst == 'sink'])
            isTable, newLabel = make_table(label, names, None)
        elif re.match(r'^\d+$', node_name):
            rec = self.recipes[node_name]
            in_ports = [ing.name for ing in rec.I]
            out_ports = [ing.name for ing in rec.O]
            isTable, newLabel = make_table(label, in_ports, out_ports)

        if isTable:
            kwargs['label'] = newLabel
            kwargs['shape'] = 'plain'
        
        graph.node(
            f'{node_name}',
            **kwargs,
            **node_style
        )

    # Populate nodes by group
    for group in groups:
        if group == 'no-group':
            # Don't draw subgraph if not part of a group
            for rec_id, kwargs in groups[group]:
                add_node_internal(g, rec_id, **kwargs)
        else:
            with g.subgraph(name=f'cluster_{group}') as c:
                self.parent_context.cLog(f'Creating subgraph {group}')
                cluster_color = self.getUniqueColor(group)

                # Populate nodes
                for rec_id, kwargs in groups[group]:
                    add_node_internal(c, rec_id, **kwargs)

                payload = group.upper()
                ln = f'<tr><td align="left"><font color="{cluster_color}" face="{self.graph_config["GROUP_FONT"]}">{payload}</font></td></tr>'
                tb = f'<<table border="0">{ln}</table>>'
                c.attr(
                    color=cluster_color,
                    label=tb,
                    fontsize=f'{self.graph_config["GROUP_FONTSIZE"]}pt'
                )

    inPort = self.getInputPortSide()
    outPort = self.getOutputPortSide()
    
    is_inverted = self.graph_config['ORIENTATION'] in ['BT', 'RL']
    is_vertical = self.graph_config['ORIENTATION'] in ['TB', 'BT']

    for io_info, edge_data in self.edges.items():
        src_node, dst_node, ing_name = io_info
        ing_quant, kwargs = edge_data['quant'], edge_data['kwargs']

        ing_id = self.getIngId(ing_name)
        quant_label = self.getQuantLabel(ing_id, ing_quant)
        # ing_label = self.getIngLabel(ing_name)

        # Strip bad arguments
        if 'locked' in kwargs:
            del kwargs['locked']

        # Assign ing color if it doesn't already exist
        ing_color = self.getUniqueColor(ing_id)

        src_has_port = self.nodeHasPort(src_node)
        dst_has_port = self.nodeHasPort(dst_node)

        src_port_name = self.getPortId(ing_name, 'o')
        dst_port_name = self.getPortId(ing_name, 'i')

        src_port = f'{src_node}:{src_port_name}' if src_has_port else src_node
        dst_port = f'{dst_node}:{dst_port_name}' if dst_has_port else dst_node

        src_port = f'{src_port}:{outPort}' if src_has_port else src_port
        dst_port = f'{dst_port}:{inPort}' if dst_has_port else dst_port

        port_style = dict(edge_style)
        
        angle = 60 if is_vertical else 20
        dist = 2.5 if is_vertical else 4
        port_style.update(labeldistance=str(dist), labelangle=str(angle))

        lab = f'({quant_label})'
        if dst_has_port:
            debugHead = ''
            if 'debugHead' in edge_data:
                debugHead = f'\n{edge_data["debugHead"]}'
            port_style.update(arrowhead='normal')
            port_style.update(headlabel=f'{lab}{debugHead}')
        if src_has_port:
            debugTail = ''
            if 'debugTail' in edge_data:
                debugTail = f'\n{edge_data["debugTail"]}'
            port_style.update(arrowtail='tee')
            port_style.update(taillabel=f'{lab}{debugTail}')

        src_is_joint_i = re.match('^joint_i', src_node)
        dst_is_joint_i = re.match('^joint_i', dst_node)
        src_is_joint_o = re.match('^joint_o', src_node)
        dst_is_joint_o = re.match('^joint_o', dst_node)
        
        #if src_is_joint_o:
        #    port_style.update(taillabel=f'{lab}')
        if src_has_port and dst_is_joint_o:
            port_style.update(headlabel=f'{lab}')
        if src_is_joint_i and dst_has_port:
            port_style.update(taillabel=f'{lab}')
        #if dst_is_joint_i:
        #    port_style.update(headlabel=f'{lab}')

        def mulcolor(h, f):
            h = h.lstrip('#')
            r,g,b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            r = max(0,min(255,int(r * f)))
            g = max(0,min(255,int(g * f)))
            b = max(0,min(255,int(b * f)))
            return '#' + ''.join(hex(x)[2:].zfill(2) for x in [r,g,b])


        g.edge(
            src_port,
            dst_port,
            fontcolor=mulcolor(ing_color, 1.5),
            color=ing_color,
            **kwargs,
            **port_style
        )

    # Output final graph
    g.render(
        self.graph_name,
        'output/',
        view=self.graph_config['VIEW_ON_COMPLETION'],
        format=self.graph_config['OUTPUT_FORMAT'],
    )

    if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
        input()
    
    if self.graph_config.get('PRINT_BOTTLENECKS'):
        self.bottleneckPrint()
