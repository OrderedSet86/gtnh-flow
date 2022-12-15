import os
import shutil
from pathlib import Path

import yaml
import json5 # Load json with comments


root = Path(__file__).parent.parent
folders_to_check = [
    root / 'projects'
]
folder_counter = 1

while folders_to_check:
    curr_folder = folders_to_check.pop()
    for path in curr_folder.iterdir():
        if path.is_dir():
            folders_to_check.append(path)
        else:
            print(path)
            if str(path).endswith('.json'):
                with open(path, 'r') as f:
                    proj_file = json5.load(f)

                post_project_path = str(path).split('projects')[-1][1:-5]

                write_name = f'{str(path)[:-5]}.yaml'
                with open(write_name, 'w') as f:
                    yaml.safe_dump(proj_file, f, default_flow_style=False, sort_keys=False)

                backup_path = f'backup/{post_project_path}.json'
                os.makedirs(Path(backup_path).parent, exist_ok=True)
                shutil.move(path, backup_path)
