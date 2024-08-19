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
            for n_e, d in [(4, 32), (4, 64), (4, 128), (5, 32), (5, 64), (5, 128)]:
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

def get_gurobi_result(gurobi_model):

    m = gurobi_model

    m.setParam("TimeLimit", 300)

    x_vars = []

    # Iterate through all variables in the Gurobi model
    for var in gurobi_model.getVars():
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

        return solution_tensor, m.ObjVal, m.Runtime

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
        for n_p in [14, 16, 18]:
            for n_e, d in [(3, 16), (3, 32), (4, 16), (4, 32)]:
                for formulation in ["sos", "bigm"]:
                    for sparsity in [0, 0.5, 0.8, 0.9]:
                        yield {
                            'data_seed': 0,
                            'training_seed': 0,
                            'n_pixel_1d': n_p,
                            'formulation': formulation,
                            'n_layers': n_e,
                            'layer_size': d,
                            'sparsity': sparsity
                        }

def run_formulation():
    for args in argument_generator():
        formulate_results = formulate(**args)
        scip = formulate_results[-1]
        scip.writeProblem("./a.mps")
        solution, obj_val, runtime = get_gurobi_result(gp.read("./a.mps"))

        dense_model, sparse_model = formulate_results[0], formulate_results[1]
        right_label, wrong_label = formulate_results[2], formulate_results[3]

        with torch.no_grad():
            print(obj_val)
            dense_result = dense_model(solution.view(1, -1))
            sparse_result = sparse_model(solution.view(1, -1))

        args["Y_max_sparse"] = obj_val
        args["Y_max_dense"] = (dense_result[0][right_label] - dense_result[0][wrong_label] + 1).item()
        args["Runtime"] = runtime


        print("Dense", obj_val, runtime)
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
