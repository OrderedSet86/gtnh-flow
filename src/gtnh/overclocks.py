# Standard libraries
import math
from bisect import bisect_right

# Pypi libraries
import yaml
from termcolor import colored

# Internal libraries
from src.data.basicTypes import Ingredient, IngredientCollection


def require(recipe, requirements):
    # requirements should be a list of [key, type, reason]
    for req in requirements:
        key, req_type, reason = req
        pass_conditions = [key in vars(recipe), isinstance(getattr(recipe, key, None), req_type)]
        if not all(pass_conditions):
            raise RuntimeError(f'Improper config! "{recipe.machine}" requires key "{key}" - it is used for {reason}.')


class OverclockHandler:


    def __init__(self, parent_context):
        self.parent_context = parent_context
        self.ignore_underclock = False # Whether to throw an error or actually underclock if
                                       # USER_VOLTAGE < EUT

        with open('data/overclock_data.yaml', 'r') as f:
            self.overclock_data = yaml.safe_load(f)

        self.voltages = self.overclock_data['voltage_data']['tiers']
        self.voltage_cutoffs = [32*pow(4, x) + 1 for x in range(len(self.voltages))]


    def modifyGTpp(self, recipe):
        if recipe.machine not in self.overclock_data['GTpp_stats']:
            raise RuntimeError('Missing OC data for GT++ multi - add to gtnhClasses/overclocks.py:GTpp_stats')

        # Get per-machine boosts
        SPEED_BOOST, EU_DISCOUNT, PARALLELS_PER_TIER = self.overclock_data['GTpp_stats'][recipe.machine]
        SPEED_BOOST = 1/(SPEED_BOOST+1)

        # Calculate base parallel count and clip time to 1 tick
        available_eut = self.voltage_cutoffs[self.voltages.index(recipe.user_voltage)]
        MAX_PARALLEL = (self.voltages.index(recipe.user_voltage) + 1) * PARALLELS_PER_TIER
        NEW_RECIPE_TIME = max(recipe.dur * SPEED_BOOST, 1)

        # Calculate current EU/t spend
        x = recipe.eut * EU_DISCOUNT
        y = min(int(available_eut/x), MAX_PARALLEL)
        recipe.parallel = y
        TOTAL_EUT = x*y

        # Debug info
        self.parent_context.log.debug(colored('Base GT++ OC stats:', 'yellow'))
        self.parent_context.log.debug(colored(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow'))

        # Attempt to GT OC the entire parallel set until no energy is left
        while TOTAL_EUT < available_eut:
            OC_EUT = TOTAL_EUT * 4
            OC_DUR = NEW_RECIPE_TIME / 2
            if OC_EUT <= available_eut:
                if OC_DUR < 1:
                    break
                self.parent_context.log.debug(colored('OC to', 'yellow'))
                self.parent_context.log.debug(colored(f'{OC_EUT=} {OC_DUR=}', 'yellow'))
                TOTAL_EUT = OC_EUT
                NEW_RECIPE_TIME = OC_DUR
            else:
                break

        recipe.eut = TOTAL_EUT
        recipe.dur = NEW_RECIPE_TIME
        recipe.I *= y
        recipe.O *= y

        return recipe


    def modifyGTppSetParallel(self, recipe, MAX_PARALLEL, speed_per_tier=1):
        available_eut = self.voltage_cutoffs[self.voltages.index(recipe.user_voltage)]

        x = recipe.eut
        y = min(int(available_eut/x), MAX_PARALLEL)
        recipe.parallel = y
        TOTAL_EUT = x*y
        NEW_RECIPE_TIME = round(recipe.dur * (speed_per_tier)**(self.voltages.index(recipe.user_voltage) + 1), 2)

        self.parent_context.log.debug(colored('Base GT++ OC stats:', 'yellow'))
        self.parent_context.log.debug(colored(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow'))

        while TOTAL_EUT < available_eut:
            OC_EUT = TOTAL_EUT * 4
            OC_DUR = NEW_RECIPE_TIME / 2
            if OC_EUT <= available_eut:
                if OC_DUR < 20:
                    break
                self.parent_context.log.debug(colored('OC to', 'yellow'))
                self.parent_context.log.debug(colored(f'{OC_EUT=} {OC_DUR=}', 'yellow'))
                TOTAL_EUT = OC_EUT
                NEW_RECIPE_TIME = OC_DUR
            else:
                break

        recipe.eut = TOTAL_EUT
        recipe.dur = NEW_RECIPE_TIME
        recipe.I *= y
        recipe.O *= y

        return recipe


    def modifyChemPlant(self, recipe):
        require(
            recipe,
            [
                ['coils', str, 'calculating recipe duration (eg "nichrome").'],
                ['pipe_casings', str, 'calculating throughput multiplier (eg "steel").']
                # ['catalyst', str, 'calculating input costs'] # TODO: No requirement until we have all catalysts
            ]
        )

        chem_plant_pipe_casings = self.overclock_data['pipe_casings']
        if recipe.pipe_casings not in chem_plant_pipe_casings:
            raise RuntimeError(f'Expected chem pipe casings in {list(chem_plant_pipe_casings)}\ngot "{recipe.pipe_casings}". (More are allowed, I just haven\'t added them yet.)')

        throughput_multiplier = chem_plant_pipe_casings[recipe.pipe_casings]
        coil_multiplier = self.overclock_data['coil_multipliers'][recipe.coils]

        # Add catalyst
        known_catalysts = {
            # Just put the actual value here, the rest is automatically calculated
            '': None,
            'orange metal catalyst': IngredientCollection(*[
                Ingredient('vanadium dust', 5),
                Ingredient('palladium dust', 5),
            ]),
            # TODO: Add other catalysts
        }

        # assert 'catalyst' in known_catalysts, f'Unknown catalyst "{recipe.catalyst}", should be in\n{known_catalysts}' # TODO: No requirements until we have all catalysts

        if hasattr(recipe, 'catalyst'):
            if recipe.catalyst == '':
                pass
            else:
                if coil_multiplier < 5.5 or recipe.pipe_casings != 'tungstensteel':
                    catalyst_cost = known_catalysts[recipe.catalyst]
                    catalyst_cost *= 1/50 # 50 durability per catalyst
                    catalyst_cost *= 1 - throughput_multiplier / 10 # 20% chance of no damage per pipe casing tier
                    recipe.I += known_catalysts[recipe.catalyst]

        recipe.dur /= coil_multiplier
        recipe.I *= throughput_multiplier
        recipe.O *= throughput_multiplier

        recipe = self.modifyStandard(recipe)

        return recipe


    def modifyZhuhai(self, recipe):
        recipe = self.modifyStandard(recipe)
        parallel_count = (self.voltages.index(recipe.user_voltage) + 2)*2
        recipe.O *= parallel_count
        return recipe


    def modifyEBF(self, recipe):
        require(
            recipe,
            [
                ['coils', str, 'calculating heat and perfect OCs for recipes (eg "nichrome").'],
                ['heat', int, 'calculating perfect OCs and heat requirement (eg "4300").'],
            ]
        )
        base_voltage = bisect_right(self.voltage_cutoffs, recipe.eut)
        user_voltage = self.voltages.index(recipe.user_voltage)
        oc_count = user_voltage - base_voltage

        actual_heat = self.overclock_data['coil_heat'][recipe.coils] + 100 * min(0, user_voltage - 2)
        excess_heat = actual_heat - recipe.heat
        eut_discount = 0.95 ** (excess_heat // 900)
        perfect_ocs = (excess_heat // 1800)

        recipe.eut = recipe.eut * 4**oc_count * eut_discount
        recipe.dur = recipe.dur / 2**oc_count / 2**max(min(perfect_ocs, oc_count), 0)

        return recipe


    def modifyPyrolyse(self, recipe):
        require(
            recipe,
            [
                ['coils', str, 'calculating recipe time (eg "nichrome").']
            ]
        )
        oc_count = self.calculateStandardOC(recipe)
        recipe.eut = recipe.eut * 4**oc_count
        recipe.dur = recipe.dur / 2**oc_count / self.overclock_data['coil_multipliers'][recipe.coils]

        return recipe


    def modifyMultiSmelter(self, recipe):
        recipe.eut = 4
        recipe.dur = 500
        recipe = self.modifyStandard(recipe)
        coil_tiering = {name: int(multiplier*2)-1 for name, multiplier in self.overclock_data['coil_multipliers'].items()}
        batch_size = 8 * 2**max(4, coil_tiering[recipe.coils])
        recipe.I *= batch_size
        recipe.O *= batch_size
        return recipe


    def modifyTGS(self, recipe):
        require(
            recipe,
            [
                ['saw_type', str, 'calculating throughput of TGS (eg "saw", "buzzsaw").']
            ]
        )
        saw_multipliers = {
            'saw': 1,
            'buzzsaw': 2,
            'chainsaw': 4,
        }
        assert recipe.saw_type in saw_multipliers, f'"saw_type" must be in {saw_multipliers}'

        oc_idx = self.voltages.index(recipe.user_voltage)
        tTier = oc_idx + 1
        TGS_base_output = (2*(tTier**2) - (2*tTier) + 5) * 5
        TGS_wood_out = TGS_base_output * saw_multipliers[recipe.saw_type]

        assert len(recipe.O) <= 1, 'Automatic TGS overclocking only supported for single output - ask dev to support saplings'

        # Mutate final recipe
        if len(recipe.O) == 0:
            recipe.O = IngredientCollection(Ingredient('wood', TGS_wood_out))
        else:
            recipe.O = IngredientCollection(Ingredient(recipe.O._ings[0].name, TGS_wood_out))
        recipe.eut = self.voltage_cutoffs[oc_idx] - 1
        print(oc_idx)
        recipe.dur = max(100/(2**(oc_idx)), 1)

        return recipe


    def modifyUtupu(self, recipe):
        require(
            recipe,
            [
                ['coils', str, 'calculating heat and perfect OCs for recipes (eg "nichrome").'],
                ['heat', int, 'calculating heat and perfect OCs for recipes (eg "4300").'],
            ]
        )

        ### First do parallel step of GTpp
        if recipe.machine not in self.overclock_data['GTpp_stats']:
            raise RuntimeError('Missing OC data for GT++ multi - add to gtnhClasses/overclocks.py:GTpp_stats')

        # Get per-machine boosts
        SPEED_BOOST, EU_DISCOUNT, PARALLELS_PER_TIER = self.overclock_data['GTpp_stats'][recipe.machine]
        SPEED_BOOST = 1/(SPEED_BOOST+1)

        # Calculate base parallel count and clip time to 1 tick
        available_eut = self.voltage_cutoffs[self.voltages.index(recipe.user_voltage)]
        MAX_PARALLEL = (self.voltages.index(recipe.user_voltage) + 1) * PARALLELS_PER_TIER
        NEW_RECIPE_TIME = max(recipe.dur * SPEED_BOOST, 1)

        # Calculate current EU/t spend
        x = recipe.eut * EU_DISCOUNT
        y = min(int(available_eut/x), MAX_PARALLEL)
        TOTAL_EUT = x*y

        # Debug info
        self.parent_context.log.debug(colored('Base GT++ OC stats:', 'yellow'))
        self.parent_context.log.debug(colored(f'{available_eut=} {MAX_PARALLEL=} {NEW_RECIPE_TIME=} {TOTAL_EUT=} {y=}', 'yellow'))

        ### Now do GT EBF OC
        base_voltage = bisect_right(self.voltage_cutoffs, TOTAL_EUT)
        user_voltage = self.voltages.index(recipe.user_voltage)
        oc_count = user_voltage - base_voltage

        actual_heat = self.overclock_data['coil_heat'][recipe.coils] # + 100 * min(0, user_voltage - 1) # I assume there's no bonus heat on UT
        excess_heat = actual_heat - recipe.heat
        eut_discount = 0.95 ** (excess_heat // 900)
        perfect_ocs = (excess_heat // 1800)

        recipe.eut = TOTAL_EUT * 4**oc_count * eut_discount
        recipe.dur = NEW_RECIPE_TIME / 2**oc_count / 2**max(min(perfect_ocs, oc_count), 0)
        recipe.I *= y
        recipe.O *= y

        return recipe


    def modifyFusion(self, recipe):
        # Ignore "tier" and just use "mk" argument for OCs
        # start is also in "mk" notation
        require(
            recipe,
            [
                ['mk', int, 'overclocking fusion. mk = actual mark run at, start = base mk. (eg mk=3, start=2)'],
                ['start', int, 'overclocking fusion. mk = actual mark run at, start = base mk. (eg mk=3, start=2)'],
            ]
        )

        mk_oc = recipe.mk - recipe.start

        bonus = 1
        if recipe.mk == 4 and mk_oc > 0:
            bonus = 2
        recipe.eut = recipe.eut * (2**mk_oc * bonus)
        recipe.dur = recipe.dur / (2**mk_oc * bonus)
        recipe.user_voltage = self.voltages[bisect_right(self.voltage_cutoffs, recipe.eut)]
        recipe.machine = f'MK{recipe.mk} {recipe.machine}'
        return recipe


    def modifyTurbine(self, recipe, fuel_type):
        require(
            recipe,
            [
                ['material', str, 'calculating power output (eg "infinity").'],
                ['size', str, 'calculating power output (eg "large").'],
            ]
        )

        fuel = recipe.I._ings[0].name
        material = recipe.material.lower()
        size = recipe.size.lower()

        with open('data/turbine_data.yaml', 'r') as f:
            turbine_data = yaml.safe_load(f)
        assert fuel in turbine_data[fuel_type], f'Unsupported fuel "{fuel}"'
        assert material in turbine_data['materials'], f'Unsupported material "{material}"'
        assert size in turbine_data['rotor_size'], f'Unsupported size "{size}"'

        material_data = turbine_data['materials'][material]

        # TODO: For now, assume optimal eut/flow as calculated on spreadsheet
        if getattr(recipe, 'flow', None) is None:
            # Calculate optimal gas flow for turbine (in EU/t)
            optimal_eut = (
                material_data['mining speed']
                    * turbine_data['rotor_size'][size]['multiplier']
                    * 50
            )
            efficiency = (
                (material_data['tier'] * 10) 
                    + 100 
                    + turbine_data['rotor_size'][size]['efficiency']
            )

            burn_value = turbine_data[fuel_type][fuel]
            optimal_flow_L_t = math.floor(optimal_eut / burn_value)
            output_eut = math.floor(optimal_flow_L_t * burn_value * efficiency / 100)
        else:
            raise NotImplementedError('Specifying "flow" feature not implemented yet')
        
        # print(f'{optimal_eut=}')
        # print(f'{optimal_flow_L_t=}')
        # print(f'{efficiency=}')
        # print(f'{output_eut=}')

        additional = []
        if fuel_type == 'steam_fuels':
            additional.append(Ingredient('[recycle] distilled water', optimal_flow_L_t//160))

        recipe.eut = 0
        recipe.dur = 1
        recipe.I._ings[0].quant = optimal_flow_L_t
        recipe.O = IngredientCollection(*[
            Ingredient('EU', output_eut),
            *additional
        ])
        recipe.efficiency = f'{efficiency}%'

        return recipe


    def modifyAAL(self, recipe):
        # Huge approximation, will not be accurate for laser overclock energy cost
        parallel_count = len(recipe.I)
        recipe.I *= parallel_count
        recipe.O *= parallel_count

        return self.modifyStandard(recipe)


    def modifyXT(self, recipe, fuel_type):
        recipe = self.modifyTurbine(recipe, fuel_type)
        recipe.I *= 16
        recipe.O *= 16

        return recipe
    

    def modifyMega(self, recipe, baseModifierFunction):
        recipe = baseModifierFunction(recipe)
        recipe.I *= 256
        recipe.O *= 256
        recipe.eut *= 256

        return recipe


    def calculateStandardOC(self, recipe):
        base_voltage = bisect_right(self.voltage_cutoffs, recipe.eut)
        user_voltage = self.voltages.index(recipe.user_voltage)
        oc_count = user_voltage - base_voltage
        if oc_count < 0:
            raise RuntimeError(f'Recipe has negative overclock! Min voltage is {base_voltage}, given OC voltage is {user_voltage}.\n{recipe}')
        return oc_count


    def modifyStandard(self, recipe):
        oc_count = self.calculateStandardOC(recipe)
        recipe.eut = recipe.eut * 4**oc_count
        recipe.dur = recipe.dur / 2**oc_count
        return recipe


    def modifyPerfect(self, recipe):
        oc_count = self.calculateStandardOC(recipe)
        recipe.eut = recipe.eut * 4**oc_count
        recipe.dur = recipe.dur / 4**oc_count
        return recipe


    def getOverclockFunction(self, recipe):
        machine_overrides = {
            # GT multis
            'pyrolyse oven': self.modifyPyrolyse,
            'large chemical reactor': self.modifyPerfect,
            'electric blast furnace': self.modifyEBF,
            'multi smelter': self.modifyMultiSmelter,
            'circuit assembly line': self.modifyPerfect,
            'fusion reactor': self.modifyFusion,
            
            'advanced assline': self.modifyAAL,
            'advanced assembling line': self.modifyAAL,
            'advanced assembly line': self.modifyAAL,
            'AAL': self.modifyAAL,

            'large gas turbine': lambda recipe: self.modifyTurbine(recipe, 'gas_fuels'),
            'XL Turbo Gas Turbine': lambda recipe: self.modifyXT(recipe, 'gas_fuels'),

            'large steam turbine': lambda recipe: self.modifyTurbine(recipe, 'steam_fuels'),
            'XL Turbo Steam Turbine': lambda recipe: self.modifyXT(recipe, 'steam_fuels'),

            # Megas
            'mega blast furnace': lambda recipe: self.modifyMega(recipe, self.modifyEBF),
            'MBF': lambda recipe: self.modifyMega(recipe, self.modifyEBF),
            'mega large chemical reactor': lambda recipe: self.modifyMega(recipe, self.modifyPerfect),
            'mega chemical reactor': lambda recipe: self.modifyMega(recipe, self.modifyPerfect),
            'MCR': lambda recipe: self.modifyMega(recipe, self.modifyPerfect),
            'mega distillation tower': lambda recipe: self.modifyMega(recipe, self.modifyStandard),
            'MDT': lambda recipe: self.modifyMega(recipe, self.modifyStandard),
            'mega vacuum freezer': lambda recipe: self.modifyMega(recipe, self.modifyStandard),
            'MVF': lambda recipe: self.modifyMega(recipe, self.modifyStandard),

            # Basic GT++ multis
            'industrial centrifuge': self.modifyGTpp,
            'industrial material press': self.modifyGTpp,
            'industrial electrolyzer': self.modifyGTpp,
            'maceration stack': self.modifyGTpp,
            'wire factory': self.modifyGTpp,
            'industrial mixing machine': self.modifyGTpp,
            'industrial sifter': self.modifyGTpp,
            'large sifter': self.modifyGTpp,
            'large thermal refinery': self.modifyGTpp,
            'industrial wash plant': self.modifyGTpp,
            'ore washing plant': self.modifyGTpp,
            'industrial extrusion machine': self.modifyGTpp,
            'large processing factory': self.modifyGTpp,
            'industrial arc furnace': self.modifyGTpp,
            'large scale auto-assembler': self.modifyGTpp,
            'cutting factory controller': self.modifyGTpp,
            'boldarnator': self.modifyGTpp,
            'dangote - distillery': self.modifyGTpp,
            'thermic heating device': self.modifyGTpp,
            'thermic heater': self.modifyGTpp,
            'industrial fluid heater': self.modifyGTpp,
            'volcanus': self.modifyGTpp,

            # Special GT++ multis
            'industrial coke oven': lambda recipe: self.modifyGTppSetParallel(recipe, 24, speed_per_tier=0.96),
            'dangote - distillation tower': lambda recipe: self.modifyGTppSetParallel(recipe, 12),
            'dangote': lambda recipe: self.modifyGTppSetParallel(recipe, 12),
            'chemical plant': self.modifyChemPlant,
            'chem plant': self.modifyChemPlant,
            'exxonmobil chemical plant': self.modifyChemPlant,
            'zhuhai': self.modifyZhuhai,
            'tree growth simulator': self.modifyTGS,
            'industrial dehydrator': self.modifyUtupu,
            'flotation cell regulator': self.modifyPerfect,
            'isamill grinding machine': self.modifyPerfect,
        }

        if recipe.machine in machine_overrides:
            return machine_overrides[recipe.machine](recipe)
        else:
            return self.modifyStandard(recipe)


    def overclockRecipe(self, recipe, ignore_underclock=False):
        ### Modifies recipe according to overclocks
        # By the time that the recipe arrives here, it should have a "user_voltage" argument which indicates
        # what the user is actually providing.
        self.ignore_underclock = ignore_underclock

        if getattr(recipe, 'do_not_overclock', False):
            return recipe

        return self.getOverclockFunction(recipe)
