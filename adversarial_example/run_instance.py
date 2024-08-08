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

    d_s = 0
    t_s = 0
    for n_p in [14, 16, 18]:
        for n_e, d in [(3, 16), (3, 32), (4, 16), (4, 32)]:
            model = train(
                data_seed=d_s,
                training_seed=t_s,
                n_pixel_1d=n_p,
                layer_size=d,
                n_layers=n_e,
                test=True,
                build_only=True,
            )

            torch.save(model.state_dict(), f"./models/mnist_{n_p}_{n_e}_{d}_{d_s}_{t_s}_dense.pth")

            for sparsity in [0.5, 0.8, 0.9]:
                sparse_model = copy.deepcopy(model)
                sparse_model = prune(sparse_model, n_p, sparsity)
                torch.save(sparse_model.state_dict(), f"./models/mnist_{n_p}_{n_e}_{d}_{d_s}_{t_s}_{sparsity}.pth")

def get_gurobi_result(gurobi_model):

    m = gurobi_model

    m.setParam("TimeLimit", 300)

    x_vars = []

    # Iterate through all variables in the Gurobi model
    for var in gurobi_model.getVars():
        if var.varName.startswith('x_'):
            x_vars.append(var)

    m.optimize()

    if m.status == 4:
        print(f'Model is infeasible or unbounded')
        return None, m.Runtime

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:
        for var in x_vars:
            match = re.match(r'x_(\d+)_(\d+)', var.varName)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                if i not in sol_dict:
                    sol_dict[i] = {}
                sol_dict[i][j] = var.x
        return m.ObjVal, m.Runtime

    elif m.status == gp.GRB.INTERRUPTED or m.status == gp.GRB.TIME_LIMIT:
        return m.ObjVal, m.Runtime

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
        scip = formulate(**args)
        scip.writeProblem("./a.mps")
        obj_val, runtime = get_gurobi_result(gp.read("./a.mps"))

        args["Y_max"] = obj_val
        args["Runtime"] = runtime

        print("Dense", obj_val, runtime)
        add_line_to_csv("./test.csv", args)

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
