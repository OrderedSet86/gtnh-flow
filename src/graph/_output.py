import re
from io import StringIO
from collections import defaultdict
from typing import Any, Iterable, Literal, Union

import graphviz
from termcolor import colored

from src.data.basicTypes import EdgeIndexType


def addNodeInternal(
        self,
        g: graphviz.Digraph,
        node_name: str,
        **kwargs
    ) -> None:

    node_style = {
        'style': 'filled',
        'fontname': self.graph_config['GENERAL_FONT'],
        'fontsize': str(self.graph_config['NODE_FONTSIZE']),
    }

    label = kwargs['label'] if 'label' in kwargs else None
    isTable = False
    newLabel = None

    def unique(sequence: Iterable) -> list: # TODO: Not sure why this exists when set could be used
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    if node_name == 'source':
        names = unique([name for src, _, name in self.edges.keys() if src == 'source'])
        isTable, newLabel = makeNodeTable(self, label, [], names)
    elif node_name == 'sink':
        names = unique([name for _, dst, name in self.edges.keys() if dst == 'sink'])
        isTable, newLabel = makeNodeTable(self, label, names, [])
    elif re.match(r'^\d+$', node_name):
        rec = self.recipes[node_name]
        in_ports = [ing.name for ing in rec.I]
        in_quants = [ing.quant for ing in rec.I]
        out_ports = [ing.name for ing in rec.O]
        out_quants = [ing.quant for ing in rec.O]
        isTable, newLabel = makeNodeTable(self, label, in_ports, out_ports, in_quants, out_quants)

    if isTable:
        kwargs['label'] = newLabel
        kwargs['shape'] = 'plain'

    g.node(
        f'{node_name}',
        **kwargs,
        **node_style
    )


def makeNodeTable(
        self,
        lab: str,
        inputs: list[str],
        outputs: list[str],
        input_quants: list[float] = None,
        output_quants: list[float] = None,
    ) -> tuple[bool, str]:
    is_inverted = self.graph_config['ORIENTATION'] in ['BT', 'RL']
    is_vertical = self.graph_config['ORIENTATION'] in ['TB', 'BT']
    num_inputs = len(inputs)
    num_outputs = len(outputs)
    has_input = num_inputs > 0
    has_output = num_outputs > 0

    if not has_input and not has_output:
        return (False, lab)

    machine_cell = ['<br />'.join(lab.split('\n'))]
    lines = [
        ('i', inputs, input_quants),
        (None, machine_cell, None),
        ('o', outputs, output_quants)
    ]
    if is_inverted:
        lines.reverse()
    lines = [(x, y, z) for x, y, z in lines if y]

    io = StringIO()
    if is_vertical:
        # Each Row is a table
        io.write('<<table border="0" cellspacing="0">')
        for port_type, line, quants in lines:
            io.write('<tr>')
            io.write('<td>')
            io.write('<table border="0" cellspacing="0">')
            io.write('<tr>')
            for i, cell in enumerate(line):
                constructCell(self, cell, i, io, port_type, quants)
            io.write('</tr>')
            io.write('</table>')
            io.write('</td>')
            io.write('</tr>')
        io.write('</table>>')
    else:
        # Each columns is a table
        io.write('<<table border="0" cellspacing="0">')
        io.write('<tr>')
        for port_type, line, quants in lines:
            io.write('<td>')
            io.write('<table border="0" cellspacing="0">')
            for i, cell in enumerate(line):
                io.write('<tr>')
                constructCell(self, cell, i, io, port_type, quants)
                io.write('</tr>')
            io.write('</table>')
            io.write('</td>')
        io.write('</tr>')
        io.write('</table>>')
    return (True, io.getvalue())


# TODO: Needs nomenclature update
def constructCell(
        self,
        cell: str,
        i: int,
        io: StringIO,
        port_type: Literal['i', 'o', None],
        quant_recipe: list[float],
    ) -> None:
    background_color = self.graph_config['BACKGROUND_COLOR']

    if port_type:
        port_id = self.getPortId(cell, port_type)
        ing_name = self.getIngLabel(cell)
        label = self.stripBrackets(ing_name)
        if quant_recipe:
            quant = self.userAccurate(quant_recipe[i])
            label = f'{label} x{quant}'

        io.write(f'<td border="1" PORT="{port_id}" bgcolor="{background_color}" color="white"><font color="white">{label}</font></td>')
    else:
        io.write(f'<td border="0">{cell}</td>')


def calculateNodeRank(self) -> int:
    # To make layout more intuitive:
    # 1. Mark all nodes with no predecessors as rank 0
    # 2. Propagate rank updates down the tree
    # 3. Send these explicit ranks to graphviz
    raise NotImplementedError()


def mulcolor(h: str, f: Union[float, int]) -> str:
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = max(0, min(255, int(r * f)))
    g = max(0, min(255, int(g * f)))
    b = max(0, min(255, int(b * f)))
    return '#' + ''.join(hex(x)[2:].zfill(2) for x in [r, g, b])


def constructPortAwareEdgeStyling(
        self,
        io_info: EdgeIndexType,
        edge_data: dict[str, Any], # TODO: Make nicer type for this
        edge_style: dict[str, str],
        is_vertical: bool,
        src_has_port: bool,
        dst_has_port: bool,
    ) -> dict[str, str]:
    src_node, dst_node, ing_name = io_info
    ing_quant = edge_data['quant']
    ing_id = self.getIngId(ing_name)
    quant_label = self.getQuantLabel(ing_id, ing_quant)

    port_style = dict(edge_style)

    # Place edge quant in a location it is likely to be readable
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

    # if src_is_joint_o:
    #    port_style.update(taillabel=f'{lab}')
    if src_has_port and dst_is_joint_o:
        port_style.update(headlabel=f'{lab}')
    if src_is_joint_i and dst_has_port:
        port_style.update(taillabel=f'{lab}')
    # if dst_is_joint_i:
    #    port_style.update(headlabel=f'{lab}')

    return port_style


def outputGraphviz(self) -> None:
    # Outputs a graphviz png using the graph info
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
        strict=False,  # Prevents edge grouping
        graph_attr={
            'bgcolor': self.graph_config['BACKGROUND_COLOR'],
            'splines': self.graph_config['LINE_STYLE'],
            'rankdir': self.graph_config['ORIENTATION'],
            'ranksep': self.graph_config['RANKSEP'],
            'nodesep': self.graph_config['NODESEP'],
            # 'newrank': True,
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

    # Populate nodes by group
    for group in groups:
        if group == 'no-group':
            # Don't draw subgraph if not part of a group
            for rec_id, kwargs in groups[group]:
                addNodeInternal(self, g, rec_id, **kwargs)
        else:
            # Draw subgraph/cluster if part of group
            with g.subgraph(name=f'cluster_{group}') as c:
                self.parent_context.log.debug(colored(f'Creating subgraph {group}'))
                cluster_color = self.getUniqueColor(group)

                # Populate nodes
                for rec_id, kwargs in groups[group]:
                    addNodeInternal(self, c, rec_id, **kwargs)

                # Make border around cluster
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
        # ing_label = self.getIngLabel(ing_name)

        # Assign ing color if it doesn't already exist
        ing_color = self.getUniqueColor(ing_id)

        # Figure out ID of connected node - add port if exist
        src_has_port = self.nodeHasPort(src_node)
        dst_has_port = self.nodeHasPort(dst_node)
        src_port_name = self.getPortId(ing_name, 'o')
        dst_port_name = self.getPortId(ing_name, 'i')

        src_port = f'{src_node}:{src_port_name}' if src_has_port else src_node
        dst_port = f'{dst_node}:{dst_port_name}' if dst_has_port else dst_node
        src_port = f'{src_port}:{outPort}' if src_has_port else src_port
        dst_port = f'{dst_port}:{inPort}' if dst_has_port else dst_port

        port_style = constructPortAwareEdgeStyling(
            self,
            io_info,
            edge_data,
            edge_style,
            is_vertical,
            src_has_port,
            dst_has_port,
        )

        color_style = dict(
            fontcolor=mulcolor(ing_color, 1.5),
            color=ing_color,
        )

        # Strip bad arguments
        if 'locked' in kwargs:
            del kwargs['locked']

        g.edge(
            src_port,
            dst_port,
            **kwargs,
            **color_style,
            **port_style,
        )

    # Output final graph
    g.render(
        filename=self.graph_name,
        directory='output/',
        view=self.graph_config['VIEW_ON_COMPLETION'],
        format=self.graph_config['OUTPUT_FORMAT'],
        engine='dot',
    )

    if self.graph_config.get('DEBUG_SHOW_EVERY_STEP', False):
        input()
