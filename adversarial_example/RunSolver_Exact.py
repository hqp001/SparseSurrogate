import os
import time
import torch
import gurobipy as gp
from formulation import formulate
from formulation_double import formulate_double

import manager

TIME_LIMIT = 300

def gurobi_callback(model, where):
    #global first_negative_time, most_negative_solution, start_time

    if where == gp.GRB.Callback.MIPSOL:


        m = model

        max_obj, max_tensor = activate_subsolver(model)
        if max_obj != None:
            m._most_negative_solution = compare_sol(m._most_negative_solution, max_tensor, m._dense_model, m._right_label, m._wrong_label)
        if m._first_negative_time == None and m._most_negative_solution < 1:
            print("First Negative Time: ", m._first_negative_time)
            m._first_negative_time = model.cbGet(gp.GRB.Callback.RUNTIME)


def activate_subsolver(model):
    binary_sol = model.cbGetSolution(model._binary)

    expr = 0

    new_constr = []

    aux_model = model._aux_model

    for var, val in zip(model._binary, binary_sol):

        sparse_var_name = "sparse_" + var.VarName
        sparse_var = aux_model.getVarByName(sparse_var_name)

        if val <= 0.5:
            new_constr.append(aux_model.addConstr(sparse_var == 0))
            expr += var
        else:
            new_constr.append(aux_model.addConstr(sparse_var == 1))
            expr += 1 - var

    # Because RunTime var is not updated yet so we need to record it ourself
    aux_model_time_limit = max(min(TIME_LIMIT - (time.time() - model._start_time), 50), 1)
    print(time.time() - model._start_time, aux_model_time_limit)
    aux_model.setParam("TimeLimit", aux_model_time_limit)
    aux_model.optimize()

    objective_value = None
    max_tensor = None

    if aux_model.status == gp.GRB.OPTIMAL or aux_model.SolCount > 0:

        objective_value = aux_model.ObjVal

        x_sol = [v.X for v in aux_model._x_vars]

        sol_dict = {}

        for var, val in zip(aux_model._x_vars, x_sol):
            # Extract row and column indices from the variable name
            _, i, j = var.VarName.split('_')
            i, j = int(i), int(j)
            if i not in sol_dict:
                sol_dict[i] = {}
            sol_dict[i][j] = val

        solution_values = [list(inner_dict.values()) for inner_dict in sol_dict.values()]
        solution_tensor = torch.tensor(solution_values)

        print("New objective value", objective_value)

        max_tensor = solution_tensor

    else:
        print("Cant fint any sol")


    for constr in new_constr:
        aux_model.remove(constr)

    model.cbLazy(expr >= 1)

    return objective_value, max_tensor


def remove_current_solution(model):
    binary_sol = model.cbGetSolution(model._binary)

    expr = 0

    for var, val in zip(model._binary, binary_sol):

        if val <= 0.5:
            expr += var
        else:
            expr += 1 - var

    model.cbLazy(expr >= 1)

    print("Removed Solution")

    return model

def compare_sol(best_sol, solution_tensor, dense_model, right_label, wrong_label):

    with torch.no_grad():
        dense_result = dense_model(solution_tensor.view(1, -1))
        dense_result = (dense_result[0][right_label] - dense_result[0][wrong_label] + 1).item()

    if best_sol == None:
        return dense_result

    elif dense_result < best_sol:
        return dense_result

    else:
        return best_sol

def get_gurobi_result(time_limit, gurobi_sparse_model, gurobi_double_model, dense_model, right_label, wrong_label):
    #global first_negative_time, most_negative_solution, start_time


    m = gurobi_sparse_model

    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", 8)
    m.setParam("LazyConstraints", 1)

    m._aux_model = gurobi_double_model
    m._aux_model.setParam("TimeLimit", 50)
    m._aux_model._x_vars = [v for v in m._aux_model.getVars() if v.VarName.startswith('x_')]
    m._dense_model = dense_model
    m._right_label = right_label
    m._wrong_label = wrong_label
    m._first_negative_time = None
    m._most_negative_solution = float('inf')
    m._x_vars = [v for v in m.getVars() if v.VarName.startswith('x_')]

    all_vars = m.getVars()
    binary_vars = [v for v in all_vars if v.VType == gp.GRB.BINARY]
    m._binary = binary_vars

    m._start_time = time.time()
    m.optimize(gurobi_callback)

    total_time = time.time() - m._start_time

    objective_value = None

    if m.status == 4 or m.status == 3:
        print(f'Model is infeasible or unbounded')

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:

        objective_value = m.ObjVal

    elif m.status == gp.GRB.INTERRUPTED or m.status == gp.GRB.TIME_LIMIT:
        objective_value = m.ObjVal if m.SolCount > 0 else None

    else:
        raise ValueError(f"Unexpected status : {m.status}")

    return objective_value, m._most_negative_solution, m._first_negative_time, m.RunTime, m.status
    #return objective_value, runtime, most_negative_solution, first_negative_time

def run_formulation():
    inputs_dict = manager.get_all_input_arguments()
    print("Retrieved Input Arguments:")
    for input_id, input_args in inputs_dict.items():
        print(f"ID: {input_id}, Arguments: {input_args}")

        args_list = ['data_seed', 'training_seed', 'n_pixel_1d',
                'n_layers', 'layer_size']
        formulate_args = manager.filter_arguments(input_args, args_list)
        formulate_args["model_path"] = f"./models/{input_id}/dense/dense.pth"

        for model_name in manager.get_all_model_names(input_id=input_id):

            print(f"\nSolving {input_id} - {model_name}")

            model_path, model_info = manager.get_model_info(input_id=input_id, model_name=model_name)
            formulate_args["surrogate_path"] = model_path

            formulate_sparse_results = formulate(**formulate_args)
            formulate_doube_results = formulate_double(**formulate_args)

            scip_sparse = formulate_sparse_results[-1]
            scip_sparse.writeProblem("./tmp_sparse.mps")
            dense_model, sparse_model = formulate_sparse_results[0], formulate_sparse_results[1]
            right_label, wrong_label = formulate_sparse_results[2], formulate_sparse_results[3]

            scip_double = formulate_doube_results[-1]
            scip_double.writeProblem("./tmp_double.mps")

            objective_value, most_negative_solution, first_negative_time, runtime, solver_status = get_gurobi_result(TIME_LIMIT, gp.read("./tmp_sparse.mps"), gp.read("./tmp_double.mps"), dense_model, right_label, wrong_label)

            gurobi_args = {}
            gurobi_args["ObjectiveValue"] = objective_value
            gurobi_args["MostNegativeValue"] = most_negative_solution
            gurobi_args["TimeFirstNegative"] = first_negative_time
            gurobi_args["SolverStatus"] = solver_status
            gurobi_args["Runtime"] = runtime

            model_info["Exact"] = gurobi_args

            manager.update_model_info(input_id=input_id, model_name=model_name, new_data=model_info)

            print(model_info)


def main():
    run_formulation()

if __name__ == "__main__":
    main()
