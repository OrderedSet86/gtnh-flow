import logging
import os
from pathlib import Path

import pytest
import yaml
from termcolor import colored

from src.data.loadMachines import recipesFromConfig
from src.graph._solver import systemOfEquationsSolverGraphGen


# Just compile and generate graph for every project
# (Minus a few whitelisted long exceptions like nanocircuits)


class ProgramContext:
    DEFAULT_CONFIG_PATH = 'tests/sanity_config.yaml'

    def __init__(self):
        self.load_graph_config(ProgramContext.DEFAULT_CONFIG_PATH)
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
                def filter(self, record):
                    record.pathname = record.pathname.replace(os.getcwd(),"")
                    return True
            handler.addFilter(PackagePathFilter())
        self.log.addHandler(handler)

        self.graph_gen = systemOfEquationsSolverGraphGen

    def load_graph_config(self, config_path):
        with open(config_path, 'r') as f:
            self.graph_config = yaml.safe_load(f)


def generateProjectPaths():
    skip_names = [
        # Too big
        'circuits/nanocircuits.yaml',

        # Not a YAML file
        'renewables/README.md',
    ]
    skip_names = [f'projects/{x}' for x in skip_names]

    skip_dirs = [
        # Incomplete
        'in_progress',

        # Caught by other tests
        'testProjects',
    ]
    skip_dirs = [f'projects/{x}' for x in skip_dirs]

    project_paths = []

    root_dir = Path('projects/')
    queue = [root_dir]
    while queue:
        dir_path = queue.pop()
        if str(dir_path) in skip_dirs:
            continue
        
        children = os.listdir(dir_path)
        for child in children:
            child = dir_path / child
            if child.is_dir():
                queue.append(child)
            else:
                if str(child) not in skip_names:
                    project_paths.append(str(child))

    return project_paths


@pytest.mark.parametrize("project_name", generateProjectPaths())
def test_lazyGenerateGraphs(project_name):
    pc = ProgramContext()
    recipes = recipesFromConfig(project_name, project_folder='')

    if project_name.endswith('.yaml'):
        project_name = project_name[:-5]

    try:
        pc.graph_gen(pc, project_name, recipes, pc.graph_config)
        assert True == True
    except Exception as e:
        assert True == False, f'Failed on {project_name} with error {e}'


if __name__ == '__main__':
    for p in generateProjectPaths():
        print(p)