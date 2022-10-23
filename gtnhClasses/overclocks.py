from bisect import bisect_right

from termcolor import cprint

from dataClasses.base import Ingredient, IngredientCollection


coil_multipliers = {
    'cupronickel': 0.5,
    'kanthal': 1.0,
    'nichrome': 1.5,
    'tungstensteel': 2.0,
    'HSS-G': 2.5,
    'HSS-S': 3.0,
    'naquadah': 3.5,
    'naquadah alloy': 4,
    'trinium': 4.5,
    'electrum flux': 5,
    'awakened draconium': 5.5,
    'infinity': 6,
    'hypogen': 6.5,
    'eternal': 7
}
coil_heat = {
    'cupronickel': 1801,
    'kanthal': 2701,
    'nichrome': 3601,
    'tungstensteel': 4501,
    'HSS-G': 5401,
    'HSS-S': 6301,
    'naquadah': 7201,
    'naquadah alloy': 8101,
    'trinium': 9001,
    'electrum flux': 9901,
    'awakened draconium': 10801,
    'infinity': 11701,
    'hypogen': 12601,
    'eternal': 13501
}

# chem plant stuff
pipe_casings = {
    'bronze': 2,
    'steel': 4,
    'titanium': 6,
    'tungstensteel': 8
}

# [speed X, EU/t discount, parallels per tier]
GTpp_stats = {
    # In order of search query:
    # machine type.*\[gt\+\+\]
    'industrial centrifuge': [1.25, 0.9, 6],
    'industrial material press': [5.0, 1.0, 4],
    'industrial electrolyzer': [1.8, 0.9, 2],
    'maceration stack': [0.6, 1.0, 8],
    'wire factory': [2.0, 0.75, 4],

    'industrial mixing machine': [2.5, 1.0, 8],
    'industrial mixer': [2.5, 1.0, 8],

    'industrial sifter': [4, 0.75, 4],

    'large thermal refinery': [1.5, 0.8, 8],
    'industrial thermal centrifuge': [1.5, 0.8, 8],

    'industrial wash plant': [4.0, 1.0, 4],
    'industrial extrusion machine': [2.5, 1.0, 4],

    'large processing factory': [2.5, 0.8, 2],
    'LPF': [2.5, 0.8, 2],

    'high current industrial arc furnace': [2.5, 1.0, 8],
    'industrial arc furnace': [2.5, 1.0, 8],

    'large scale auto-assembler': [2.0, 1.0, 2],
    'cutting factory controller': [2.0, 0.75, 4],

    'boldarnator': [2.0, 0.75, 8],
    'industrial rock breaker': [2.0, 0.75, 8],

    'dangote - distillery': [0, 1.0, 48],

    'utupu-tanuri': [1.2, 0.5, 4],
    'utupu tanuri': [1.2, 0.5, 4],
    'industrial dehydrator': [1.2, 0.5, 4],
}

voltages = ['LV', 'MV', 'HV', 'EV', 'IV', 'LuV', 'ZPM', 'UV', 'UHV', 'UEV', 'UIV', 'UMV', 'UXV']
voltage_cutoffs = [32*pow(4, x) + 1 for x in range(len(voltages))]


def require(recipe, requirements):
    # requirements should be a list of [key, type]
    for req in requirements:
        key, req_type = req
        pass_conditions = [key in vars(recipe), isinstance(getattr(recipe, key), req_type)]
        if not all(pass_conditions):
            raise RuntimeError(f'Improper config! Ensure {recipe.machine} has key {key} of type {req_type}.')


def modifyGTpp(recipe):
    if recipe.machine not in GTpp_stats:
        raise RuntimeError('Missing OC data for GT++ multi - add to gtnhClasses/overclocks.py:GTpp_stats')

    # Get per-machine boosts
    SPEED_BOOST, EU_DISCOUNT, PARALLELS_PER_TIER = GTpp_stats[recipe.machine]
    SPEED_BOOST = 1/(SPEED_BOOST+1)

    # Calculate base parallel count and clip time to 1 tick
    available_eut = voltage_cutoffs[voltages.index(recipe.user_voltage)]
    MAX_PARALLEL = (voltages.index(recipe.user_voltage) + 1) * PARALLELS_PER_TIER
    NEW_RECIPE_TIME = max(recipe.dur * SPEED_BOOST, 1)

    # Calculate current EU/t spend
    x = recipe.eut * EU_DISCOUNT
    y = min(int(available_eut/x), MAX_PARALLEL)
    TOTAL_EUT = x*y

    # Debug info
    cprint('Base GT++ OC stats:', 'yellow')
    cprint(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow')

    # Attempt to GT OC the entire parallel set until no energy is left
    while TOTAL_EUT < available_eut:
        OC_EUT = TOTAL_EUT * 4
        OC_DUR = NEW_RECIPE_TIME / 2
        if OC_EUT <= available_eut:
            if OC_DUR < 20:
                break
            cprint('OC to', 'yellow')
            cprint(f'{OC_EUT=} {OC_DUR=}', 'yellow')
            TOTAL_EUT = OC_EUT
            NEW_RECIPE_TIME = OC_DUR
        else:
            break

    recipe.eut = TOTAL_EUT
    recipe.dur = NEW_RECIPE_TIME
    recipe.I *= y
    recipe.O *= y

    return recipe


def modifyGTppSetParallel(recipe, MAX_PARALLEL, speed_per_tier=1):
    available_eut = voltage_cutoffs[voltages.index(recipe.user_voltage)]

    x = recipe.eut
    y = min(int(available_eut/x), MAX_PARALLEL)
    TOTAL_EUT = x*y
    NEW_RECIPE_TIME = round(recipe.dur * (speed_per_tier)**(voltages.index(recipe.user_voltage) + 1), 2)

    cprint('Base GT++ OC stats:', 'yellow')
    cprint(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow')

    while TOTAL_EUT < available_eut:
        OC_EUT = TOTAL_EUT * 4
        OC_DUR = NEW_RECIPE_TIME / 2
        if OC_EUT <= available_eut:
            if OC_DUR < 20:
                break
            cprint('OC to', 'yellow')
            cprint(f'{OC_EUT=} {OC_DUR=}', 'yellow')
            TOTAL_EUT = OC_EUT
            NEW_RECIPE_TIME = OC_DUR
        else:
            break

    recipe.eut = TOTAL_EUT
    recipe.dur = NEW_RECIPE_TIME
    recipe.I *= y
    recipe.O *= y

    return recipe


def modifyChemPlant(recipe):
    assert 'coils' in dir(recipe), 'Chem plant requires "coils" argument (eg "nichrome")'
    assert 'pipe_casings' in dir(recipe), 'Chem plant requires "pipe_casings" argument (eg "steel")'
    # assert 'solid_casings' in dir(recipe), 'Chem plant requires "solid_casings" argument (eg "vigorous laurenium")'

    chem_plant_pipe_casings = {
        'bronze': 1,
        'steel': 2,
        'titanium': 3,
        'tungstensteel': 4,
    }
    if recipe.pipe_casings not in chem_plant_pipe_casings:
        raise RuntimeError(f'Expected chem pipe casings in {list(chem_plant_pipe_casings)}\ngot "{recipe.pipe_casings}"')

    recipe.dur /= coil_multipliers[recipe.coils]
    throughput_multiplier = (2*chem_plant_pipe_casings[recipe.pipe_casings])
    recipe.I *= throughput_multiplier
    recipe.O *= throughput_multiplier

    recipe = modifyStandard(recipe)

    return recipe


def modifyZhuhai(recipe):
    recipe = modifyStandard(recipe)
    parallel_count = (voltages.index(recipe.user_voltage) + 2)*2
    recipe.O *= parallel_count
    return recipe


def modifyEBF(recipe):
    require(
        recipe,
        [
            ['coils', str],
            ['heat', int],
        ]
    )
    base_voltage = bisect_right(voltage_cutoffs, recipe.eut)
    user_voltage = voltages.index(recipe.user_voltage)
    oc_count = user_voltage - base_voltage

    actual_heat = coil_heat[recipe.coils] + 100 * min(0, user_voltage - 2)
    excess_heat = actual_heat - recipe.heat
    eut_discount = 0.95 ** (excess_heat // 900)
    perfect_ocs = (excess_heat // 1800)

    recipe.eut = recipe.eut * 4**oc_count * eut_discount
    recipe.dur = recipe.dur / 2**oc_count / 2**max(min(perfect_ocs, oc_count), 0)

    return recipe


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


def modifyMultiSmelter(recipe):
    recipe.eut = 4
    recipe.dur = 500
    recipe = modifyStandard(recipe)
    coil_list = list(coil_multipliers)
    batch_size = 8 * 2**max(4, coil_list.index(recipe.coils))
    recipe.I *= batch_size
    recipe.O *= batch_size
    return recipe


def modifyTGS(recipe):
    # This enforces a known overclock chart - too lazy to look up the code
    require(
        recipe,
        [
            ['saw_type', str]
        ]
    )
    saw_multipliers = {
        'saw': 1,
        'buzzsaw': 2,
        'chainsaw': 4,
    }
    assert recipe.saw_type in saw_multipliers, f'"saw_type" must be in {saw_multipliers}'
    assert recipe.user_voltage in {'LV', 'MV', 'HV', 'EV', 'IV', 'LuV', 'ZPM'}, 'Bother Order#0001 on Discord and tell him to read the code and stop being lazy'

    TGS_outputs = [5, 9, 17, 29, 45, 65, 89]
    oc_idx = voltages.index(recipe.user_voltage)
    TGS_wood_out = TGS_outputs[oc_idx] * saw_multipliers[recipe.saw_type]

    assert len(recipe.O) <= 1, 'Automatic TGS overclocking only supported for single output - contact dev for more details'

    # Mutate final recipe
    if len(recipe.O) == 0:
        recipe.O = IngredientCollection(Ingredient('wood', TGS_wood_out))
    else:
        recipe.O = IngredientCollection(Ingredient(recipe.O._ings[0].name, TGS_wood_out))
    recipe.eut = voltage_cutoffs[oc_idx] - 1
    print(oc_idx)
    recipe.dur = max(100/(2**(oc_idx)), 1)

    return recipe


def modifyUtupu(recipe):
    require(
        recipe,
        [
            ['coils', str],
            ['heat', int],
        ]
    )

    ### First do parallel step of GTpp
    if recipe.machine not in GTpp_stats:
        raise RuntimeError('Missing OC data for GT++ multi - add to gtnhClasses/overclocks.py:GTpp_stats')

    # Get per-machine boosts
    SPEED_BOOST, EU_DISCOUNT, PARALLELS_PER_TIER = GTpp_stats[recipe.machine]
    SPEED_BOOST = 1/(SPEED_BOOST+1)

    # Calculate base parallel count and clip time to 1 tick
    available_eut = voltage_cutoffs[voltages.index(recipe.user_voltage)]
    MAX_PARALLEL = (voltages.index(recipe.user_voltage) + 1) * PARALLELS_PER_TIER
    NEW_RECIPE_TIME = max(recipe.dur * SPEED_BOOST, 1)

    # Calculate current EU/t spend
    x = recipe.eut * EU_DISCOUNT
    y = min(int(available_eut/x), MAX_PARALLEL)
    TOTAL_EUT = x*y

    # Debug info
    cprint('Base GT++ OC stats:', 'yellow')
    cprint(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow')

    ### Now do GT EBF OC
    base_voltage = bisect_right(voltage_cutoffs, TOTAL_EUT)
    user_voltage = voltages.index(recipe.user_voltage)
    oc_count = user_voltage - base_voltage

    actual_heat = coil_heat[recipe.coils] # + 100 * min(0, user_voltage - 1) # I assume there's no bonus heat on UT
    excess_heat = actual_heat - recipe.heat
    eut_discount = 0.95 ** (excess_heat // 900)
    perfect_ocs = (excess_heat // 1800)

    recipe.eut = TOTAL_EUT * 4**oc_count * eut_discount
    recipe.dur = NEW_RECIPE_TIME / 2**oc_count / 2**max(min(perfect_ocs, oc_count), 0)
    recipe.I *= y
    recipe.O *= y

    return recipe


def modifyFusion(recipe):
    # Ignore "tier" and just use "mk" argument for OCs
    # start is also in "mk" notation
    require(
        recipe,
        [
            ['mk', int],
            ['start', int],
        ]
    )

    mk_oc = recipe.mk - recipe.start

    bonus = 1
    if recipe.mk == 4 and mk_oc > 0:
        bonus = 2
    recipe.eut = recipe.eut * (2**mk_oc * bonus)
    recipe.dur = recipe.dur / (2**mk_oc * bonus)
    recipe.user_voltage = voltages[bisect_right(voltage_cutoffs, recipe.eut)]
    recipe.machine = f'MK{recipe.mk} {recipe.machine}'
    return recipe


def calculateStandardOC(recipe):
    base_voltage = bisect_right(voltage_cutoffs, recipe.eut)
    user_voltage = voltages.index(recipe.user_voltage)
    oc_count = user_voltage - base_voltage
    if oc_count < 0:
        raise RuntimeError(f'Recipe has negative overclock! Min voltage is {base_voltage}, given OC voltage is {user_voltage}.\n{recipe}')
    return oc_count


def modifyStandard(recipe):
    oc_count = calculateStandardOC(recipe)
    recipe.eut = recipe.eut * 4**oc_count
    recipe.dur = recipe.dur / 2**oc_count
    return recipe


def modifyPerfect(recipe):
    oc_count = calculateStandardOC(recipe)
    recipe.eut = recipe.eut * 4**oc_count
    recipe.dur = recipe.dur / 4**oc_count
    return recipe


def overclockRecipe(recipe):
    ### Modifies recipe according to overclocks
    # By the time that the recipe arrives here, it should have a "user_voltage" argument which indicates
    # what the user is actually providing.
    machine_overrides = {
        # GT multis
        'pyrolyse oven': modifyPyrolyse,
        'large chemical reactor': modifyPerfect,
        'LCR': modifyPerfect,
        'electric blast furnace': modifyEBF,
        'EBF': modifyEBF,
        'blast furnace': modifyEBF,
        'multi smelter': modifyMultiSmelter,
        'circuit assembly line': modifyPerfect,
        'CAL': modifyPerfect,
        'fusion': modifyFusion,
        'fusion reactor': modifyFusion,

        # Basic GT++ multis
        'industrial centrifuge': modifyGTpp,
        'industrial material press': modifyGTpp,
        'industrial electrolyzer': modifyGTpp,
        'maceration stack': modifyGTpp,
        'wire factory': modifyGTpp,
        'industrial mixing machine': modifyGTpp,
        'industrial mixer': modifyGTpp,
        'industrial sifter': modifyGTpp,
        'large thermal refinery': modifyGTpp,
        'industrial thermal centrifuge': modifyGTpp,
        'industrial wash plant': modifyGTpp,
        'industrial extrusion machine': modifyGTpp,
        'large processing factory': modifyGTpp,
        'LPF': modifyGTpp,
        'high current industrial arc furnace': modifyGTpp,
        'industrial arc furnace': modifyGTpp,
        'large scale auto-assembler': modifyGTpp,
        'cutting factory controller': modifyGTpp,
        'boldarnator': modifyGTpp,
        'industrial rock breaker': modifyGTpp,
        'dangote - distillery': modifyGTpp,

        # Special GT++ multis
        'industrial coke oven': lambda recipe: modifyGTppSetParallel(recipe, 24, speed_per_tier=0.96),
        'ICO': lambda recipe: modifyGTppSetParallel(recipe, 24, speed_per_tier=0.96),
        'dangote - distillation tower': lambda recipe: modifyGTppSetParallel(recipe, 12),
        'chem plant': modifyChemPlant,
        'chemical plant': modifyChemPlant,
        'exxonmobil': modifyChemPlant,
        'zhuhai': modifyZhuhai,
        'tree growth simulator': modifyTGS,
        'tgs': modifyTGS,
        'utupu-tanuri': modifyUtupu,
        'utupu tanuri': modifyUtupu,
        'industrial dehydrator': modifyUtupu,
        'flotation cell': modifyPerfect,
        'flotation cell regulator': modifyPerfect,
        'isamill': modifyPerfect,
        'isamill grinding machine': modifyPerfect,
    }
    if recipe.machine in machine_overrides:
        return machine_overrides[recipe.machine](recipe)
    else:
        return modifyStandard(recipe)
