from dataclasses import dataclass
from typing import Dict


@dataclass
class Ingredient:
    name: str
    quant: float


class IngredientCollection:
    def __init__(self, *ingredient_list):
        self._ings = ingredient_list
        self._ingdict = {x.name: x.quant for x in self._ings}

    def __iter__(self):
        return iter(self._ings)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._ings[idx]
        elif isinstance(idx, str):
            return self._ingdict[idx]
        else:
            raise RuntimeError(f'Improper access to {self} using {idx}')


class Recipe:
    def __init__(
            self,
            machine_name,
            inputs,
            outputs,
            eut,
            dur,
            **kwargs
        ):
        self.machine = machine_name
        self.I = inputs
        self.O = outputs
        self.eut = eut
        self.dur = dur
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == '__main__':
    r = Recipe(
        'centrifuge',
        IngredientCollection(
            Ingredient('glass dust', 1)
        ),
        IngredientCollection(
            Ingredient('silicon dioxide', 1)
        ),
        5,
        80
    )
    print(r)
    print(r.I['glass dust'])
    print(list(r.I))