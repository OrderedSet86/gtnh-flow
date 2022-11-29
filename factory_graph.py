# Standard libraries
import logging
import os
from pathlib import Path

# Pypi libraries
import yaml
from termcolor import colored, cprint

# Internal libraries
from prototypes.linearSolver import systemOfEquationsSolverGraphGen
from src.graph import Graph
from src.data.loadMachines import recipesFromConfig

# Conditional imports based on OS
try: # Linux
    import readline
except Exception: # Windows
    import pyreadline3 as readline


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

    
    def run(self, graph_gen=None):
        if graph_gen == None:
            graph_gen = self.standardGraphGen

        with open('config_factory_graph.yaml', 'r') as f:
            graph_config = yaml.safe_load(f)
        
        if graph_config['USE_NEW_SOLVER']:
            graph_gen = systemOfEquationsSolverGraphGen
        
        # Set up autcompletion config
        projects_path = Path('projects')
        readline.parse_and_bind('tab: complete')
        readline.set_completer_delims('')

        while True:
            def completer(text, state):
                prefix = ''
                suffix = text
                if '/' in text:
                    parts = text.split('/')
                    prefix = '/'.join(parts[:-1])
                    suffix = parts[-1]

                target_path = projects_path / prefix
                valid_tabcompletes = os.listdir(target_path)
                valid_completions = [x for x in valid_tabcompletes if x.startswith(suffix)]
                if state < len(valid_completions): # Only 1 match
                    completion = valid_completions[state]
                    if prefix != '':
                        completion = ''.join([prefix, '/', completion])
                    if not completion.endswith('.yaml'):
                        completion += '/'
                    return completion
                else:
                    return None

            readline.set_completer(completer)

            cprint('Please enter project path (example: "power/oil/light_fuel.yaml", tab autocomplete allowed)', 'blue')
            project_name = input(colored('> ', 'green'))

            recipes = recipesFromConfig(project_name)

            if project_name.endswith('.yaml'):
                project_name = project_name[:-5]

            graph_gen(self, project_name, recipes, graph_config)


    @staticmethod
    def standardGraphGen(self, project_name, recipes, graph_config):
        # Create graph and render
        g = Graph(project_name, recipes, self, graph_config=graph_config)
        g.connectGraph()
        g.balanceGraph()
        g.outputGraphviz()


if __name__ == '__main__':
    pc = ProgramContext()
    pc.run()