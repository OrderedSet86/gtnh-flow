import math
import re
from collections import defaultdict
from typing import Literal


# This file is for the "port" style nodes
# as designed by Usagirei in https://github.com/OrderedSet86/gtnh-flow/pull/4


def stripBrackets(self, ing: str) -> str:
    if self.graph_config['STRIP_BRACKETS']:
        prefix = False
        if ing[:2] == '\u2588 ':
            prefix = True
        stripped = ing.split(']')[-1].strip()
        if prefix and stripped[:2] != '\u2588 ': 
            stripped = '\u2588 ' + stripped
        return stripped
    else:
        return ing


def nodeHasPort(self, node: str) -> bool:
    if node in ['source', 'sink']:
        return True
    if re.match(r'^\d+$', node):
        return True
    return False


def getOutputPortSide(self) -> str:
    dir = self.graph_config['ORIENTATION']
    if dir == 'TB':
        return 's'
    elif dir == 'BT':
        return 'n'
    elif dir == 'LR':
        return 'e'
    else:
        return 'w'


def getInputPortSide(self) -> str:
    dir = self.graph_config['ORIENTATION']
    if dir == 'TB':
        return 'n'
    elif dir == 'BT':
        return 's'
    elif dir == 'LR':
        return 'w'
    else:
        return 'e'


def getUniqueColor(self, port_id: str) -> str:
    if port_id not in self._color_dict:
        self._color_dict[port_id] = next(self._color_cycler)
    return self._color_dict[port_id]


def getPortId(
        self,
        ing_name: str,
        port_type: Literal['i', 'o']
    ) -> str:
    normal = re.sub(' ','_', ing_name).lower().strip()
    return f'{port_type}_{normal}'


def getIngId(self, ing_name: str) -> str:
    id = ing_name
    id = re.sub(r'\[.*?\]', '', id)
    id = id.strip()
    id = re.sub(r' ', '_', id)
    return id.lower()


def getIngLabel(self, ing_name: str) -> str:
    capitalization_exceptions = {
        'eu': 'EU',
    }
    ing_id = self.getIngId(ing_name)
    if ing_id in capitalization_exceptions:
        return capitalization_exceptions[ing_id]
    else:
        return ing_name.title()


def getQuantLabel(self, ing_id: str, ing_quant: float) -> str:
    unit_exceptions = {
        'eu': lambda eu: f'{int(math.floor(eu / 20))}/t'
    }
    if ing_id in unit_exceptions:
        return unit_exceptions[ing_id](ing_quant)
    else:
        return f'{self.userRound(ing_quant)}/s'


def _combineOutputs(self) -> None:
    ings = defaultdict(list)
    for src,dst,ing in self.edges.keys():
        ings[(src,ing)].append(dst)
    merge = {k:v for k,v in ings.items() if len(v) > 1}

    n = 0
    for t,lst in merge.items():
        src,ing = t
        
        joint_id = f'joint_o_{n}'
        n = n+1

        ing_id = self.getIngId(ing)
        ing_color = self.getUniqueColor(ing_id)
        self.addNode(joint_id, shape='point', color=ing_color)
        qSum = 0
        for dst in lst:
            k = (src,dst,ing)
            info = self.edges[k]
            self.edges.pop(k)
            quant = info['quant']
            kwargs = info['kwargs']
            qSum = qSum + quant
            self.addEdge(joint_id, dst, ing, quant, **kwargs)
        
        self.addEdge(src, joint_id, ing, qSum)


def _combineInputs(self) -> None:
    ings = defaultdict(list)
    for src,dst,ing in self.edges.keys():
        ings[(dst,ing)].append(src)
    merge = {k:v for k,v in ings.items() if len(v) > 1}

    n = 0
    for t,lst in merge.items():
        dst,ing = t
        
        joint_id = f'joint_i_{n}'
        n = n+1

        ing_id = self.getIngId(ing)
        ing_color = self.getUniqueColor(ing_id)
        self.addNode(joint_id, shape='point', color=ing_color)
        qSum = 0
        for src in lst:
            k = (src,dst,ing)
            info = self.edges[k]
            self.edges.pop(k)
            quant = info['quant']
            kwargs = info['kwargs']
            qSum = qSum + quant
            self.addEdge(src, joint_id, ing, quant, **kwargs)
        
        self.addEdge(joint_id, dst, ing, qSum)