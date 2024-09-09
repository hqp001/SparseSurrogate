import copy
import re
import sys
from train import train
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from prune import prune
from formulation import formulate
import gurobipy as gp

import csv
import os

def run_training():

    adversarial_data = []

    for d_s, t_s in [(0, 0), (1, 3), (8, 12), (90, 99)]:
        for n_p in [28, 22, 18]:
            for n_e, d in [(4, 256), (4, 512), (5, 256), (5, 512)]:
                model = train(
                    data_seed=d_s,
                    training_seed=t_s,
                    n_pixel_1d=n_p,
                    layer_size=d,
                    n_layers=n_e,
                    test=True,
                    build_only=True,
                )

                os.makedirs("./adversarial_example/models", exist_ok=True)
                torch.save(model.state_dict(), f"./adversarial_example/models/mnist_{n_p}_{n_e}_{d}_{d_s}_{t_s}_dense.pth")

                for sparsity in [0.5, 0.8, 0.9]:
                    sparse_model = copy.deepcopy(model)
                    sparse_model = prune(sparse_model, n_p, sparsity)
                    torch.save(sparse_model.state_dict(), f"./adversarial_example/models/mnist_{n_p}_{n_e}_{d}_{d_s}_{t_s}_{sparsity}.pth")

def remove_current_solution(model):
    # Retrieve integer variables
    int_vars = [v for v in model.getVars() if v.vType in [gp.GRB.BINARY]]

    # Get the current solution values for these variables
    current_solution = [v.x for v in int_vars]

    # Create a new constraint to remove this solution
    expr = 0
    for i, var in enumerate(int_vars):
        if current_solution[i] <= 0.5:
            expr += int_vars
        else:
            expr += 1 - int_vars

    model.addConstr(expr >= 1, name="exclude_solution")

    return model


best_sol = None

def compare_sol(solution_tensor, dense_model, right_label, wrong_label):

    global best_sol

    with torch.no_grad():
        dense_result = dense_model(solution_tensor.view(1, -1))
        dense_result = (dense_result[0][right_label] - dense_result[0][wrong_label] + 1).item()


    if best_sol == None:
        best_sol = (dense_result, solution_tensor)

    elif dense_result < best_sol[0]:
        best_sol = (dense_result, solution_tensor)


def get_gurobi_result(time_limit, gurobi_model, dense_model, right_label, wrong_label):

    if time_limit <= 0:
        return None, None, None

    m = gurobi_model

    m.reset()

    m.setParam("TimeLimit", time_limit)

    x_vars = []

    # Iterate through all variables in the Gurobi model
    for var in m.getVars():
        if var.varName.startswith('x_'):
            x_vars.append(var)

    m.optimize()

    sol_dict = {}

    if m.status == 4 or m.status == 3:
        print(f'Model is infeasible or unbounded')
        return None, None, m.Runtime

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:

        for var in x_vars:
            match = re.match(r'x_(\d+)_(\d+)', var.varName)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                if i not in sol_dict:
                    sol_dict[i] = {}
                sol_dict[i][j] = var.x

        solution_values = [list(inner_dict.values()) for inner_dict in sol_dict.values()]

        solution_tensor = torch.tensor(solution_values)

        compare_sol(solution_tensor, dense_model, right_label, wrong_label)

        time_limit -= m.Runtime

        remove_current_solution(m)
        get_gurobi_result(time_limit, m, dense_model, right_label, wrong_label)

        return None, None, None

    elif m.status == gp.GRB.INTERRUPTED or m.status == gp.GRB.TIME_LIMIT:
        return None, m.ObjVal, m.Runtime

    else:
        raise ValueError(f"Unexpected status : {m.status}")

def add_line_to_csv(file_name, data):
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            # If file doesn't exist, write the header
            writer.writerow(data.keys())

        # Write the data row
        writer.writerow(data.values())

def argument_generator():
    for d_s, t_s in [(0, 0), (1, 3), (8, 12), (90, 99)]:
        for n_p in [28, 22, 18]:
            for n_e, d in [(4, 32), (4, 64), (4, 128), (5, 32), (5, 64), (5, 128)]:
                for sparsity in [0, 0.5, 0.8, 0.9]:
                    yield {
                        'data_seed': 0,
                        'training_seed': 0,
                        'n_pixel_1d': n_p,
                        'n_layers': n_e,
                        'layer_size': d,
                        'sparsity': sparsity
                    }

def run_formulation():
    global best_sol
    for args in argument_generator():
        best_sol = None
        formulate_results = formulate(**args)
        scip = formulate_results[-1]
        scip.writeProblem("./a.mps")

        dense_model, sparse_model = formulate_results[0], formulate_results[1]
        right_label, wrong_label = formulate_results[2], formulate_results[3]

        solution, obj_val, runtime = get_gurobi_result(500, gp.read("./a.mps"), dense_model, right_label, wrong_label)

        if best_sol != None:
            args["Y_min"] = best_sol[0]

        else:
            args["Y_min"] = None

        args["Runtime"] = runtime


        add_line_to_csv("./adversarial_example/test.csv", args)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <train/solve>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "train":
        run_training()
    elif mode == "solve":
        run_formulation()

if __name__ == "__main__":
    main()
