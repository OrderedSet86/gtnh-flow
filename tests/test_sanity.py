import os
from pathlib import Path

import pytest

from factory_graph import ProgramContext
from src.data.loadMachines import recipesFromConfig


# Just compile and generate graph for every project
# (Minus a few whitelisted long exceptions like nanocircuits)


def generateProjectPaths() -> list[str]:
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
def test_lazyGenerateGraphs(project_name: str) -> None:
    pc = ProgramContext('tests/sanity_config.yaml')
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