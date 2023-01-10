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


# def generateProjectPaths():
#     return ['projects/pe/apple.yaml']


@pytest.mark.parametrize("project_name", generateProjectPaths())
def test_lazyGenerateGraphs(project_name):
    pc = ProgramContext()
    recipes = recipesFromConfig(project_name, project_folder='')

    if project_name.endswith('.yaml'):
        project_name = project_name[:-5]
    with open('tests/sanity_config.yaml', 'r') as f:
        graph_config = yaml.safe_load(f)

    try:
        systemOfEquationsSolverGraphGen(pc, project_name, recipes, graph_config)
        assert True == True
    except Exception as e:
        assert True == False, f'Failed on {project_name} with error {e}'


if __name__ == '__main__':
    for p in generateProjectPaths():
        print(p)