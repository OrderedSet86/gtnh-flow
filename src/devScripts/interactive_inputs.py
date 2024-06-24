import cvxpy as cp
from src.graph._lpProject import LpProject
from src.graph._lpScaledMatrix import LpScaledMatrix
import numpy as np
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
import time
import sys
import keyboard
from enum import Enum


class TableIO(Enum):
    IN_TABLE = 1
    OUT_TABLE = 2


class State(Enum):
    # Number order is used for sort
    WHITELISTED = 1
    WHITELISTED_AUTO = 2
    BLACKLISTED = 3
    NORMAL = 4
    # BLACKLISTED_AUTO = 5


def solveMinFreeVariables(project: LpProject, in_table, out_table):

    def var_dict_for(names, **kwargs):
        # var = cp.Variable(len(names), **kwargs) if len(names) > 0 else []
        var = cp.Variable(len(names), **kwargs)
        # Fun fact, vars is not iterable but you can get items from it by index
        return var, dict(zip(names, [var[i] for i in range(len(names))]))

    recipe_vars, recipe_vars_dict = var_dict_for(project.recipe_names, name="recipes", nonneg=True)
    in_switch, in_switch_dict = var_dict_for(project.variables, name="in_switch", boolean=True)
    in_sub, in_sub_dict = var_dict_for(project.variables, name="in_sub", bounds=[0, 1])
    out_switch, out_switch_dict = var_dict_for(project.variables, name="out_switch", boolean=True)
    out_sub, out_sub_dict = var_dict_for(project.variables, name="out_sub", bounds=[0, 1])

    # Target is the output of running one of each recipe (i.e. sum of each recipe vector)
    fake_target = np.sum(project.ing_matrix, axis=1)
    scaled_matrix = LpScaledMatrix(project.ing_matrix, fake_target)
    priority_ratio = scaled_matrix.max_min_ratio**4

    constraints = []

    # for item, recipe_coeffs in zip(project.variables, scaled_matrix.matrix):

    #     problem_eq = cp.sum(cp.multiply(recipe_vars, recipe_coeffs)) + input_term == 0
    #     constraints.append(problem_eq)

    # Item quantities must add up (for each item, item io * amount of recipe + input/output term == 0)
    constraints.append(
        cp.matmul(scaled_matrix.scaled_matrix, recipe_vars)
        + priority_ratio * (in_switch - in_sub)
        - priority_ratio * (out_switch - out_sub)
        == 0
    )
    constraints.append((in_switch - in_sub) >= 0)
    constraints.append((out_switch - out_sub) >= 0)

    def add_user_constraints(table, switch_dict):
        for item, state in table:
            if state == State.BLACKLISTED:
                constraints.append(switch_dict[item] == 0)
            elif state == State.WHITELISTED:
                constraints.append(switch_dict[item] == 1)

    add_user_constraints(in_table, in_switch_dict)
    add_user_constraints(out_table, out_switch_dict)

    # Every recipe must be used (here represented as >= 1)
    constraints.append(recipe_vars >= 1)

    objective = -1 * (cp.sum(in_sub) + cp.sum(out_sub)) + priority_ratio * (cp.sum(in_switch) + cp.sum(out_switch))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    # print("Problem Status:", problem.status, "in", problem.compilation_time, "seconds")
    if problem.status != cp.OPTIMAL:
        return False, None, None

    def update_table(table, switch_dict):
        for i, (item, state) in enumerate(table):
            if state in [State.WHITELISTED, State.BLACKLISTED]:
                continue
            if switch_dict[item].value > 0.5:
                table[i][1] = State.WHITELISTED_AUTO
            else:
                table[i][1] = State.NORMAL
        return table

    return True, update_table(in_table, in_switch_dict), update_table(out_table, out_switch_dict)


def interactiveIOChooser(project: LpProject):
    in_table = [[name, State.NORMAL] for name in project.variables]
    out_table = [[name, State.NORMAL] for name in project.variables]

    selected = [TableIO.IN_TABLE, 0]
    stop_requested = False
    selection_to_table = {TableIO.IN_TABLE: [], TableIO.OUT_TABLE: []}

    def run_auto():
        nonlocal in_table, out_table
        success, i, o = solveMinFreeVariables(project, in_table, out_table)
        if success:
            in_table, out_table = i, o
        return success

    def get_table(table):
        nonlocal in_table, out_table
        return {TableIO.IN_TABLE: in_table, TableIO.OUT_TABLE: out_table}[table]

    def move_selection(offset):
        nonlocal selected
        n = len(get_table(selected[0]))
        selected[1] = (selected[1] + offset) % n

    def get_symbol_and_style(state):
        if state == State.NORMAL:
            return " ", "white on black"
        elif state == State.BLACKLISTED:
            return "[red]✗[/red]", "black on red"
        elif state == State.WHITELISTED:
            return "[green]✓[/green]", "black on green"
        elif state == State.WHITELISTED_AUTO:
            return "A", "black on pale_green1"
        else:
            return "?", "yellow"

    def update_selected(state):
        nonlocal selected, selection_to_table
        if selection_to_table is None:
            return
        mapped_index = selection_to_table[selected[0]][selected[1]]
        entry = get_table(selected[0])[mapped_index]
        old_state = entry[1]
        new_state = State.NORMAL if entry[1] == state else state

        # Try new state and rollback if it fails
        entry[1] = new_state
        if not run_auto():
            entry[1] = old_state
            print("Auto Failed - probably, you tried to blacklist something that is required")

    def on_key_press(event):
        nonlocal selected, stop_requested
        if event.name == "q" or event.name == "ctrl+c":
            stop_requested = True
        elif event.name == "x":
            # Mark item as blacklisted
            update_selected(State.BLACKLISTED)
        elif event.name == "enter":
            # Mark item as whitelisted
            update_selected(State.WHITELISTED)
        elif event.name == "up":
            # Move selection up within the current table
            move_selection(-1)
        elif event.name == "down":
            # Move selection down within the current table
            move_selection(+1)
        elif event.name == "left" or event.name == "right":
            # Change selected table
            selected[0] = {TableIO.IN_TABLE: TableIO.OUT_TABLE, TableIO.OUT_TABLE: TableIO.IN_TABLE}[selected[0]]

    def gen_table(tableIO, title):
        nonlocal selection_to_table, selected
        table = get_table(tableIO)
        new_table_view = Table(title, "State", style="cyan", expand=True)
        sorted_items = sorted(enumerate(table), key=lambda x: (x[1][1].value, x[1][0]))
        selection_to_table[tableIO], table_items = zip(*sorted_items)
        # Flip table->sorted to get sorted->table
        # selection_to_table = np.argsort(table_argsort)

        for i, (item, state) in enumerate(table_items):
            symbol, style = get_symbol_and_style(state)
            if tableIO == selected[0] and i == selected[1]:
                style = "black on white"

            new_table_view.add_row(item, symbol, style=style)
        return new_table_view

    def refresh_table_view():
        in_table = gen_table(TableIO.IN_TABLE, "Input")
        out_table = gen_table(TableIO.OUT_TABLE, "Output")
        table_layout = Layout(name="tables")
        table_layout.split_row(in_table, out_table)
        root.update(table_layout)

    root = Layout(Text.from_markup("[yellow]Loading...[/yellow]"), name="root")
    run_auto()

    # start = time.time()
    with Live(root, refresh_per_second=4):
        while not stop_requested:
            time.sleep(0.05)

            refresh_table_view()
            keyboard.on_press(on_key_press, suppress=True)


if __name__ == "__main__":
    project_path = None
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = input("Enter project path: ")
    project = LpProject.fromConfig(project_path)
    interactiveIOChooser(project)
