import itertools
from typing import Set

from termcolor import colored

from src.gtnh.overclocks import OverclockHandler


class IdentityGroup:
    def __init__(self, ground_truth_machine: str, aliases: Set[str]):
        self.gtm = ground_truth_machine
        self.aliases = aliases
    def isCanonical(self, machine_name):
        return machine_name == self.gtm
    def isAlias(self, machine_name):
        return machine_name in self.aliases


machine_identity_groups = [
    # NOTE: If the aliases list is empty, you have to use set() instead of {}

    ['pyrolyse oven', {'pyro oven'}],
    ['large chemical reactor', {'lcr'}],
    ['electric blast furnace', {'ebf', 'blast furnace'}],
    ['multi smelter', {'smelter'}],
    ['circuit assembly line', {'cal'}],
    ['fusion reactor', {'fusion'}],
    ['advanced assline', {'advanced assembly line', 'advanced assembling line', 'aal'}],
    ['large gas turbine', {'lgt'}],
    ['xl turbo gas turbine', {'xlgt', 'xltgt'}],
    ['large steam turbine', {'lst'}],
    ['xl turbo steam turbine', {'xlst', 'xltst'}],
    ['mebf', {'mbf', 'mega blast furnace'}],
    ['mdt', {'mega distillation tower'}],
    ['mcr', {'mega chemical reactor', 'mega large chemical reactor'}],
    ['mvf', {'mega vacuum freezer'}],
    ['distillation tower', {'dt'}],

    ['industrial centrifuge', {'centrifuge++'}],
    ['industrial material press', {'bender++', 'bending++'}],
    ['industrial electrolyzer', {'electrolyzer++'}],
    ['maceration stack', {'macerator++'}],
    ['wire factory', {'wiremill++'}],
    ['industrial mixing machine', {'mixer++', 'industrial mixer'}],
    ['industrial sifter', {'sifter++', 'large sifter'}],
    ['large thermal refinery', {'thermal refinery++', 'industrial thermal centrifuge'}],
    ['industrial wash plant', {'washplant++', 'industrial washing plant', 'ore washer++', 'ore washing plant', 'owp++'}],
    ['industrial extrusion machine', {'extruder++'}],
    ['large processing factory', {'lpf'}],
    ['industrial arc furnace', {'arc furnace++', 'high current industrial arc furnace'}],
    ['large scale auto-assembler', {'lsaa', 'assembler++'}],
    ['cutting factory', {'cutting factory controller'}],
    ['boldarnator', {'rock breaker++', 'industrial rock breaker'}],
    ['dangote - distillery', {'distillery++'}],
    ['thermic heating device', {'thermic heater', 'heater++', 'industrial fluid heater'}],
    ['volcanus', {'volc', 'ebf++'}],
    ['dangote - distillation tower', {'dt++', 'dangote'}],
    ['industrial coke oven', {'ico'}],
    ['chemical plant', {'chem plant', 'exxonmobil', 'exxonmobil chemical plant'}],
    ['zhuhai', set()],
    ['tree growth simulator', {'tgs'}],
    ['industrial dehydrator', {'utupu', 'utupu-tanuri', 'utupu tanuri', 'dehydrator++'}],
    ['flotation cell regulator', {'flotation cell'}],
    ['isamill grinding machine', {'isamill'}],
]
machine_identity_groups = [IdentityGroup(name, aliases) for name, aliases in machine_identity_groups]


class Graph:


    def __init__(self, graph_name, recipes, parent_context, graph_config=None):
        self.graph_name = graph_name
        self.recipes = {str(i): x for i, x in enumerate(recipes)}
        self.nodes = {}
        self.edges = {} # uniquely defined by (machine from, machine to, ing name)
        self.parent_context = parent_context
        self.graph_config = graph_config
        if self.graph_config is None:
            self.graph_config = {}

        # Populated later on
        self.adj = None
        self.adj_machine = None

        self._color_dict = dict()
        if self.graph_config.get('USE_RAINBOW_EDGES', None):
            self._color_cycler = itertools.cycle(self.graph_config['EDGECOLOR_CYCLE'])
        else:
            self._color_cycler = itertools.cycle(['white'])

        # Convert machine names to canonical name
        for i, rec in enumerate(recipes):
            for id_group in machine_identity_groups:
                machine_name = rec.machine
                if id_group.isCanonical(machine_name):
                    pass
                elif id_group.isAlias(machine_name):
                    self.parent_context.log.info(colored(
                        f'Found "{machine_name}", converting to canonical name "{id_group.gtm}"', 'blue')
                    )
                    recipes[i].machine = id_group.gtm
                    break

        # Overclock all recipes to the provided user voltage
        oh = OverclockHandler(self.parent_context)
        for i, rec in enumerate(recipes):
            recipes[i] = oh.overclockRecipe(rec)
            rec.base_eut = rec.eut

        # DEBUG
        self.parent_context.log.debug('Recipes after overclocking:')
        for rec in recipes:
            self.parent_context.log.debug(colored(rec, 'yellow'))
        self.parent_context.log.debug('')

    # Graph utility functions
    from ._utils import (
        userRound,
        userAccurate,
        addNode,
        addEdge,
        createAdjacencyList,
        _iterateOverMachines,
        _checkIfMachine,
    )
    userRound = staticmethod(userRound)
    userAccurate = staticmethod(userAccurate)

    # Utilities for "port node" style graphviz nodes
    from ._portNodes import (
        stripBrackets,
        nodeHasPort,
        getOutputPortSide,
        getInputPortSide,
        getUniqueColor,
        getPortId,
        getIngId,
        getIngLabel,
        getQuantLabel,
        _combineInputs,
        _combineOutputs,
    )

    from ._output import (
        outputGraphviz,
    )
