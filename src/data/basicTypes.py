from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Union


@dataclass
class Ingredient:
    name: str
    quant: float


class IngredientCollection:
    def __init__(self, *ingredient_list: list[Ingredient]) -> None:
        self._ings = list(ingredient_list)
        # Note: name is not a unique identifier for multi-input situations
        # therefore, need to defaultdict a list
        self._ingdict = defaultdict(list)
        for ing in self._ings:
            self._ingdict[ing.name].append(ing.quant)

    def __iter__(self) -> Iterable[Ingredient]:
        return iter(self._ings)

    def __getitem__(self, idx: Union[int, str]) -> Union[Ingredient, list[float]]:
        # int goes to self._ings, str goes to self._ingdict
        if isinstance(idx, int):
            return self._ings[idx]
        elif isinstance(idx, str):
            return self._ingdict[idx]
        else:
            raise RuntimeError(f'Improper access to {self} using {idx}')

    def __repr__(self) -> str:
        return str([x for x in self._ings])

    def __mul__(self, mul_num: Union[float, int]) -> 'IngredientCollection':
        assert isinstance(mul_num, (int, float))
        for ing in self._ings:
            ing.quant *= mul_num
        self._ingdict = defaultdict(list)
        for ing in self._ings:
            self._ingdict[ing.name].append(ing.quant)
        # self._ingdict = {x.name: x.quant for x in self._ings}

        return self

    def __len__(self) -> int:
        return len(self._ings)

    def addItem(self, item: Ingredient) -> None:
        self._ings.append(item)
        self._ingdict[item.name].append(item.quant)

    def __add__(self, other: 'IngredientCollection') -> 'IngredientCollection':
        for ing in other:
            self.addItem(ing)
        return self

    def itemAmount(self, name: str) -> int:
        return sum(self._ingdict.get(name, []))

    def __contains__(self, item) -> bool:
        if isinstance(item, str):
            # Checks if an ingredient with the name exists
            return item in self._ingdict and len(self._ingdict[item]) > 0
        elif isinstance(item, Ingredient):
            # If this collection has the ingredient and
            # has more quantity than the ingredient has,
            # return true
            quant = item.quant
            return quant > 0 and quant <= self.itemAmount(item.name)
        return False


class Recipe:
    def __init__(
            self,
            machine_name: str,
            user_voltage: str,
            inputs: IngredientCollection,
            outputs: IngredientCollection,
            eut: float,
            dur: float, # With new subtick this is tracked as float
            **kwargs
        ) -> None:
        self.machine = machine_name
        self.user_voltage = user_voltage
        self.I = inputs
        self.O = outputs
        self.eut = eut
        self.dur = dur
        self.multiplier = -1
        self.base_eut = eut # Used for final graph output
        for key, value in kwargs.items():
            # quick fix to ignore cases
            # this implies that config files better use lower-case too
            if type(value) is str:
                value = value.lower()
            setattr(self, key, value)

    def __repr__(self) -> str:
        return str([f'{x}={getattr(self, x)}' for x in vars(self)])

    def __mul__(self, mul_num: Union[int, float]) -> 'Recipe':
        assert isinstance(mul_num, (int, float))
        assert self.multiplier == -1 # Undefined behavior with multiple multiplications

        self.I *= mul_num
        self.O *= mul_num
        self.eut *= mul_num
        self.multiplier = mul_num

        return self


EdgeIndexType = tuple[str, str, str] # (node_from, node_to, ing_name)
# EdgeDataType = dict[] # TODO: Make nicer type for this


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
