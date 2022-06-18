import json
import json5 # Load json with comments
from pathlib import Path


root = Path(__file__).parent
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

            with open(path, 'r') as f:
                proj_file = json5.load(f)

            for recipe in proj_file:
                print(recipe['dur'], end=' ')
                recipe['dur'] = round(recipe['dur'] / 20, 2)
                if int(recipe['dur']) == float(recipe['dur']):
                    recipe['dur'] = int(recipe['dur'])
                print(recipe['dur'])

            with open(path, 'w') as f:
                json.dump(proj_file, f, indent=4)
