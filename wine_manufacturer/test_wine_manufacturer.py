import os
import csv
import numpy as np
from pyscipopt import Model, quicksum
from utils import read_csv_to_dict, train_torch_neural_network

from pyscipopt_ml.add_predictor import add_predictor_constr

from prune import prune

import gurobipy as gp

def build_and_optimise_wine_manufacturer(
    data_seed=42,
    training_seed=42,
    n_vineyards=35,
    n_wines_to_produce=5,
    formulation="sos",
    n_estimators_layers=3,
    max_depth=3,
    layer_size=16,
    epsilon=0.0001,
    sparsity=0
):
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # Path to red wine data
    data_dict = read_csv_to_dict("./tests/data/red_wine_quality.csv")

    features = [
        "fixed.acidity",
        "volatile.acidity",
        "citric.acid",
        "residual.sugar",
        "chlorides",
        "free.sulfur.dioxide",
        "total.sulfur.dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    n_features = len(features)
    budget = 1.7 * n_wines_to_produce

    # Generate the actual input data arrays for the ML predictors
    X = []
    quality = np.array([float(x) for x in data_dict["quality"]]).reshape(
        -1,
    )

    for feature in features:
        X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Train the ML predictor
    reg, dataloader = train_torch_neural_network(
        X, quality, n_estimators_layers, layer_size, training_seed, reshape=True
    )

    prune(reg, dataloader, sparsity)

    # Create artificial data from some vineyards
    vineyard_order = np.arange(X.shape[0])
    data_random_state.shuffle(vineyard_order)
    vineyard_litre_limits = data_random_state.uniform(0.25, 0.35, n_vineyards)
    vineyard_costs = data_random_state.uniform(1, 2, n_vineyards)
    vineyard_features = []
    low_quality_vineyards_i = 0

    for i in vineyard_order:
        if low_quality_vineyards_i >= n_vineyards:
            break
        if quality[i] <= 5:
            low_quality_vineyards_i += 1
            vineyard_features.append(X[i])
    vineyard_features = np.array(vineyard_features)

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of each wine
    feature_vars = np.zeros((n_wines_to_produce, n_features), dtype=object)
    quality_vars = np.zeros((n_wines_to_produce, 1), dtype=object)
    wine_mixture_vars = np.zeros((n_wines_to_produce, n_vineyards), dtype=object)

    for i in range(n_wines_to_produce):
        quality_vars[i][0] = scip.addVar(vtype="C", lb=0, ub=10, name=f"quality_{i}")
        for j in range(n_features):
            max_val = np.max(X[:, j])
            min_val = np.min(X[:, j])
            lb = max(0, min_val - 0.1 * max_val)
            ub = 1.1 * max_val
            feature_vars[i][j] = scip.addVar(vtype="C", lb=lb, ub=ub, name=f"feature_{i}_{j}")
        for k in range(n_vineyards):
            wine_mixture_vars[i][k] = scip.addVar(
                vtype="C", lb=0, ub=vineyard_litre_limits[k], name=f"mixture_{i}_{k}"
            )

    # Now create constraints on the wine blending
    for i in range(n_wines_to_produce):
        for j in range(n_features):
            scip.addCons(
                feature_vars[i][j]
                == quicksum(
                    wine_mixture_vars[i][k] * vineyard_features[k][j] for k in range(n_vineyards)
                ),
                name=f"mixture_cons_{i}_{j}",
            )
    for i in range(n_wines_to_produce):
        scip.addCons(
            quicksum(wine_mixture_vars[i][k] for k in range(n_vineyards)) == 1,
            name=f"wine_mix_{i}",
        )
    for k in range(n_vineyards):
        scip.addCons(
            quicksum(wine_mixture_vars[i][k] for i in range(n_wines_to_produce))
            <= vineyard_litre_limits[k],
            name=f"vineyard_limit_{k}",
        )

    # Add the budget constraint
    scip.addCons(
        quicksum(
            quicksum(wine_mixture_vars[i][k] * vineyard_costs[k] for k in range(n_vineyards))
            for i in range(n_wines_to_produce)
        )
        <= budget,
        name=f"budget_cons",
    )

    # Add the ML constraint. Add in a single batch!
    pred_cons = add_predictor_constr(
        scip,
        reg,
        feature_vars,
        quality_vars,
        unique_naming_prefix="predictor_",
        epsilon=epsilon,
        formulation=formulation,
    )

    # Add a constraint ensuring minimum wine quality on those produced
    min_wine_quality = data_random_state.uniform(4.2, 4.5)
    for i in range(n_wines_to_produce):
        scip.addCons(quality_vars[i][0] >= min_wine_quality, name=f"min_quality_{i}")

    # Set the SCIP objective
    scip.setObjective(
        quicksum(-quality_vars[i][0] for i in range(n_wines_to_produce)) / n_wines_to_produce + 10
    )

    scip.writeProblem(
        f"./wine_manufacturer/formulation/wine_formulation.mps"
    )

    return gp.read(f"./wine_manufacturer/formulation/wine_formulation.mps")

def test_wine_manufacturer_sklearn_mlp_bigm():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=18,
        training_seed=18,
        n_vineyards=35,
        n_wines_to_produce=5,
        formulation="bigm",
        max_depth=3,
        n_estimators_layers=3,
        layer_size=7,
    )

def get_gurobi_result(gurobi_model):

    m = gurobi_model

    m.setParam("TimeLimit", 300)

    m.optimize()

    if m.status == 4 or m.status == 3:
        print(f'Model is infeasible or unbounded')
        return None, m.Runtime

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:

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
            writer.writerow(data.keys())

        writer.writerow(data.values())

def argument_generators():

    for d_s, t_s in [(0, 0), (1, 3), (5, 8)]:
        wine_data = []
        for n_v in [30, 35]:
            for n_w in [4, 5]:
                for n_e, d in [(2, 16), (2, 32), (3, 16)]:
                    for sparsity in [0, 0.5, 0.8, 0.9]:
                        yield {
                            "data_seed": d_s,
                            "training_seed": t_s,
                            "n_vineyards": n_v,
                            "n_wines_to_produce": n_w,
                            "n_estimators_layers": n_e,
                            "layer_size": d,
                            "epsilon": 0.0001,
                            "sparsity": sparsity
                        }

os.makedirs("./wine_manufacturer/formulation", exist_ok=True)
for args in argument_generators():
    gurobi_model = build_and_optimise_wine_manufacturer(**args)
    result, runtime = get_gurobi_result(gurobi_model)
    args["Result"] = result
    args["Runtime"] = runtime

    add_line_to_csv("./wine_manufacturer/result.csv", args)
