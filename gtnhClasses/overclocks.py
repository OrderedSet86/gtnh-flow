from bisect import bisect_right

from ..dataClasses.recipe import Recipe, IngredientCollection, Ingredient


coil_multipliers = {
    'cupronickel': 0.5,
    'kanthal': 1.0,
    'nichrome': 1.5,
    'tungstensteel': 2.0,
    'HSSG': 2.5,
    'HSSS': 3.0,
    'naquadah': 3.5,
    'naquadah alloy': 4,
    'trinium': 4.5,
    'electrum flux': 5,
    'awakened draconium': 5.5,
}
coil_heat = {
    'cupronickel': 1801,
    'kanthal': 2701,
    'nichrome': 3601,
    'tungstensteel': 4501,
    'HSSG': 5401,
    'HSSS': 6301,
    'naquadah': 7201,
    'naquadah alloy': 8101,
    'trinium': 9001,
    'electrum flux': 9901,
    'awakened draconium': 10801,
}

# chem plant stuff
pipe_casings = {
    'bronze': 2,
    'steel': 4,
    'titanium': 6,
    'tungstensteel': 8
}

voltage_cutoffs = [32, 128, 512, 2048, 8192, 32768, 131_072, 524_288, 2_097_152]
voltages = ['LV', 'MV', 'HV', 'EV', 'IV', 'LuV', 'ZPM', 'UV', 'UHV']


def require(recipe, requirements):
    # requirements should be a list of [key, type]
    for req in requirements:
        key, req_type = req
        pass_conditions = [key in vars(recipe), isinstance(getattr(recipe, key), req_type)]
        if not all(pass_conditions):
            raise RuntimeError(f'Improper config! Ensure {recipe.machine} has key {key} of type {req_type}.')


def modifyPyrolyse(recipe):
    require(
        recipe,
        [
            ['coils', str]
        ]
    )
    oc_count = calculateStandardOC(recipe)
    recipe.eut = recipe.eut * 4**oc_count
    recipe.dur = recipe.dur / 2**oc_count / coil_multipliers[recipe.coils]
    return recipe


def calculateStandardOC(recipe):
    base_voltage = bisect_right(voltage_cutoffs, recipe.eut)
    user_voltage = voltages.index(recipe.user_voltage)
    oc_count = user_voltage - base_voltage
    return oc_count


def modifyStandard(recipe):
    oc_count = calculateStandardOC(recipe)
    recipe.eut = recipe.eut * 4**oc_count
    recipe.dur = recipe.dur / 2**oc_count
    return recipe


def overclockRecipe(recipe):
    ### Modifies recipe according to overclocks
    # By the time that the recipe arrives here, it should have a "user_voltage" argument which indicates
    # what the user is actually providing.
    machine_overrides = {
        'pyrolyse oven': modifyPyrolyse
    }
    if recipe.machine in machine_overrides:
        return machine_overrides[recipe.machine](recipe)
    else:
        return modifyStandard(recipe)
