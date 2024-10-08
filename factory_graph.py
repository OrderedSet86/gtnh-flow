# Standard libraries
import argparse
import logging
import os
import traceback
from pathlib import Path
from typing import Literal, Union

# Pypi libraries
import yaml
from termcolor import colored, cprint

# Internal libraries
from src.data.loadMachines import recipesFromConfig
from src.graph._solver import systemOfEquationsSolverGraphGen

# Conditional imports based on OS
try: # Linux
    import readline
except Exception: # Windows
    import pyreadline3 as readline


class ProgramContext:


    def __init__(self, default_config_path: str = 'config_factory_graph.yaml') -> None:
        self.default_config_path = default_config_path
        self.load_graph_config(default_config_path)
        streamhandler_level = self.graph_config.get('STREAMHANDLER_LEVEL', 'INFO')

        self.log = logging.getLogger('flow.log')
        self.log.setLevel(logging.DEBUG)

        if streamhandler_level == 'DEBUG':
            fmtstring = '%(pathname)s:%(lineno)s %(levelname)s %(message)s'
        else:
            fmtstring = '%(filename)s:%(lineno)s %(levelname)s %(message)s'
        formatter = logging.Formatter(
            fmt=fmtstring,
            datefmt='%Y-%m-%dT%H:%M:%S%z', # ISO 8601
        )

        handler = logging.StreamHandler() # outputs to stderr
        handler.setFormatter(formatter)
        handler.setLevel(logging.getLevelName(self.graph_config.get('STREAMHANDLER_LEVEL', 'INFO')))
        if streamhandler_level == 'DEBUG':
            # https://stackoverflow.com/a/74605301
            class PackagePathFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> Literal[True]:
                    record.pathname = record.pathname.replace(os.getcwd(),"")
                    return True
            handler.addFilter(PackagePathFilter())
        self.log.addHandler(handler)

        self.graph_gen = systemOfEquationsSolverGraphGen


    def load_graph_config(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            self.graph_config = yaml.safe_load(f)


    def generate_one(self, project_name: str) -> None:
        if not project_name.endswith('.yaml'):
            raise Exception(f'Invalid project file. *.yaml file expected. Got: {project_name}.')

        recipes = recipesFromConfig(project_name)
        try:
            self.graph_gen(self, project_name[:-5], recipes, self.graph_config)
        except Exception as e:
            cprint(traceback.format_exc(), 'red')
            self.log.error(colored(f'Error generating graph for project "{project_name}": {e}', 'red'))
            self.log.error(colored('If error cause is not obvious, please notify dev: https://github.com/OrderedSet86/gtnh-flow/issues', 'red'))


    def run_noninteractive(self, projects: list[str]) -> None:
        for project_name in projects:
            self.generate_one(project_name)


    def run_interactive(self) -> None:
        # Set up autcompletion config
        projects_path = Path('projects')
        readline.parse_and_bind('tab: complete')
        readline.set_completer_delims('')

        while True:
            def completer(text: str, state: int) -> Union[str, None]:
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
            if not project_name.endswith('.yaml'):
                # Assume when the user wrote "power/fish/methane", they meant "power/fish/methane.yaml"
                # This happens because autocomplete will not add .yaml if there are alternatives (like "power/fish/methane_no_biogas")
                project_name += '.yaml'

            self.load_graph_config(self.default_config_path)
            self.generate_one(project_name)


    def run(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='config_factory_graph.yaml', help='Path to the global .yaml configuration file.')
        parser.add_argument('--interactive', action='store_true', help='Force interactive mode.')
        parser.add_argument('--graph_gen', type=str, default='systemOfEquationsSolverGraphGen', help='Type of the graph generator to use.')
        parser.add_argument('--no_view_on_completion', action='store_true', help='Override the VIEW_ON_COMPLETION config setting to False. Useful when running mass jobs.')
        parser.add_argument('projects', type=str, nargs='*', help='Paths to project files (.yaml) to be processed.')

        args = parser.parse_args()

        if args.graph_gen == 'systemOfEquationsSolverGraphGen':
            self.graph_gen = systemOfEquationsSolverGraphGen
        else:
            raise Exception('Invalid graph generator.')

        self.load_graph_config(args.config)

        if args.no_view_on_completion:
            self.graph_config['VIEW_ON_COMPLETION'] = False

        # For backwards compatibility enter interactive mode when no project is specified.
        if len(args.projects) == 0:
            args.interactive = True

        if args.interactive:
            if len(args.projects) > 0:
                raise Exception('Projects cannot be specified at this point in the interactive mode.')
            self.run_interactive()
        else:
            self.run_noninteractive(args.projects)


if __name__ == '__main__':
    pc = ProgramContext()
    pc.run()