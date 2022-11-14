from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Ingredient:
    name: str
    quant: float


class IngredientCollection:
    def __init__(self, *ingredient_list):
        self._ings = ingredient_list
        # Note: name is not a unique identifier for multi-input situations
        # therefore, need to defaultdict a list
        self._ingdict = defaultdict(list)
        for ing in self._ings:
            self._ingdict[ing.name].append(ing.quant)
        # self._ingdict = {x.name: x.quant for x in self._ings}

    def __iter__(self):
        return iter(self._ings)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._ings[idx]
        elif isinstance(idx, str):
            return self._ingdict[idx]
        else:
            raise RuntimeError(f'Improper access to {self} using {idx}')

    def __repr__(self):
        return str([x for x in self._ings])

    def __mul__(self, mul_num):
        for ing in self._ings:
            ing.quant *= mul_num
        self._ingdict = defaultdict(list)
        for ing in self._ings:
            self._ingdict[ing.name].append(ing.quant)
        # self._ingdict = {x.name: x.quant for x in self._ings}

        return self

    def __len__(self):
        return len(self._ings)


class Recipe:
    def __init__(
            self,
            machine_name,
            user_voltage,
            inputs,
            outputs,
            eut,
            dur,
            **kwargs
        ):
        self.machine = machine_name
        self.user_voltage = user_voltage
        self.I = inputs
        self.O = outputs
        self.eut = eut
        self.dur = dur
        self.multiplier = 1
        self.base_eut = eut # Used for final graph output
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str([f'{x}={getattr(self, x)}' for x in vars(self)])

    def __mul__(self, mul_num):
        assert isinstance(mul_num, (int, float))
        assert self.multiplier == 1 # Undefined behavior with multiple multiplications

        self.I *= mul_num
        self.O *= mul_num
        self.eut *= mul_num
        self.multiplier = mul_num

        return self



if __name__ == '__main__':
    r = Recipe(
        'centrifuge',
        'LV',
        IngredientCollection(
            Ingredient('glass dust', 1)
        ),
        IngredientCollection(
            Ingredient('silicon dioxide', 1)
        ),
        5,
        80,
    )
    print(r)
    print(r.I['glass dust'])
    print(list(r.I))