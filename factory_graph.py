# Standard libraries
import logging
import sys

# Pypi libraries
import yaml
from termcolor import colored

# Internal libraries
from graphClasses.graph import Graph
from dataClasses.load import recipesFromConfig


class ProgramContext:


    def __init__(self):
        logging.basicConfig(level=logging.INFO)


    @staticmethod
    def cLog(msg, color='white', level=logging.DEBUG):
        # Not sure how to level based on a variable, so just if statements for now
        if level == logging.DEBUG:
            logging.debug(colored(msg, color))
        elif level == logging.INFO:
            logging.info(colored(msg, color))
        elif level == logging.WARNING:
            logging.warning(colored(msg, color))

    
    def run(self):
        with open('config_factory_graph.yaml', 'r') as f:
            graph_config = yaml.safe_load(f)

        if len(sys.argv) > 1:
            project_name = sys.argv[1]
        else:
            raise RuntimeError('Need a project config name as an argument! Will be pulled from projects/{name}.json')

        recipes = recipesFromConfig(project_name)

        # Create graph and render
        g = Graph(project_name, recipes, self, graph_config=graph_config)
        g.connectGraph()
        g.balanceGraph()
        g.outputGraphviz()


if __name__ == '__main__':
    pc = ProgramContext()
    pc.run()