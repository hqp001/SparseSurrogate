import sys
import os
import torch
import gurobipy as gp
from formulation import formulate

import manager

TIME_LIMIT = 300
MAP_BATCH = {
    -1: -1,
    0: 0,
    1: 1,
    2: 8,
    3: 90
}
BATCH_NUM = -1

def gurobi_callback(model, where):
    #global first_negative_time, most_negative_solution, start_time

    """
    if where == gp.GRB.Callback.MIPNODE:
        obj_val = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)

        if obj_val < 0:
            current_time = time.time()
            if first_negative_time is None:
                first_negative_time = current_time - start_time

            if obj_val < most_negative_solution:
                most_negative_solution = obj_val
    """

    if where == gp.GRB.Callback.MIPSOL:
        x_vars = [v for v in model.getVars() if v.VarName.startswith('x_')]

        x_sol = model.cbGetSolution(x_vars)

        sol_dict = {}

        for var, val in zip(x_vars, x_sol):
            # Extract row and column indices from the variable name
            _, i, j = var.VarName.split('_')
            i, j = int(i), int(j)
            if i not in sol_dict:
                sol_dict[i] = {}
            sol_dict[i][j] = val


        solution_values = [list(inner_dict.values()) for inner_dict in sol_dict.values()]
        solution_tensor = torch.tensor(solution_values)

        m = model

        m._most_negative_solution = compare_sol(m._most_negative_solution, solution_tensor, m._dense_model, m._right_label, m._wrong_label)
        print("Current negative sol: ", m._most_negative_solution)
        if m._first_negative_time == None and m._most_negative_solution < 1:
            print("First Negative Time: ", m._first_negative_time)
            m._first_negative_time = model.cbGet(gp.GRB.Callback.RUNTIME)

        #remove_current_solution(model)

def remove_current_solution(model):
    int_vars = [v for v in model.getVars() if v.vType in [gp.GRB.BINARY]]

    current_solution = [v.x for v in int_vars]

    expr = 0
    for i, var in enumerate(int_vars):
        if current_solution[i] <= 0.5:
            expr += int_vars
        else:
            expr += 1 - int_vars

    model.addConstr(expr >= 1, name="exclude_solution")

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

def get_gurobi_result(time_limit, gurobi_model, dense_model, right_label, wrong_label):
    #global first_negative_time, most_negative_solution, start_time


    m = gurobi_model

    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", 8)
    #m.setParam("LazyConstraints", 1)

    m._dense_model = dense_model
    m._right_label = right_label
    m._wrong_label = wrong_label
    m._first_negative_time = None
    m._most_negative_solution = float('inf')

    m.optimize(gurobi_callback)

    objective_value = None

    if m.status == 4 or m.status == 3:
        print(f'Model is infeasible or unbounded')

    elif m.status == gp.GRB.OPTIMAL or m.SolCount > 0:

        objective_value = m.ObjVal

    elif m.status == gp.GRB.INTERRUPTED or m.status == gp.GRB.TIME_LIMIT:
        objective_value = m.ObjVal if m.SolCount > 0 else None

    else:
        raise ValueError(f"Unexpected status : {m.status}")

    return objective_value, m._most_negative_solution, m._first_negative_time, m.Runtime, m.status
    #return objective_value, runtime, most_negative_solution, first_negative_time

def add_line_to_csv(file_name, data):
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(data.keys())

        writer.writerow(data.values())

def run_formulation():
    inputs_dict = manager.get_all_input_arguments()
    print("Retrieved Input Arguments:")
    for input_id, input_args in inputs_dict.items():
        print(f"ID: {input_id}, Arguments: {input_args}")

        args_list = ['data_seed', 'training_seed', 'n_pixel_1d',
                'n_layers', 'layer_size']

        if input_args['data_seed'] != MAP_BATCH[BATCH_NUM]:
            continue

        formulate_args = manager.filter_arguments(input_args, args_list)
        formulate_args["model_path"] = f"./models/{input_id}/dense/dense.pth"

        for model_name in manager.get_all_model_names(input_id=input_id):

            print(f"\nSolving {input_id} - {model_name}")

            model_path, model_info = manager.get_model_info(input_id=input_id, model_name=model_name)
            formulate_args["surrogate_path"] = model_path

            formulate_results = formulate(**formulate_args)

            scip = formulate_results[-1]
            scip.writeProblem(f"./{BATCH_NUM}.mps")
            dense_model, sparse_model = formulate_results[0], formulate_results[1]
            right_label, wrong_label = formulate_results[2], formulate_results[3]

            objective_value, most_negative_solution, first_negative_time, runtime, solver_status = get_gurobi_result(TIME_LIMIT, gp.read(f"./{BATCH_NUM}.mps"), dense_model, right_label, wrong_label)

            gurobi_args = {}
            gurobi_args["ObjectiveValue"] = objective_value
            gurobi_args["MostNegativeValue"] = most_negative_solution
            gurobi_args["TimeFirstNegative"] = first_negative_time
            gurobi_args["SolverStatus"] = solver_status
            gurobi_args["Runtime"] = runtime
            gurobi_args["Batch_num"] = BATCH_NUM

            model_info["NonCallBack"] = gurobi_args

            manager.update_model_info(input_id=input_id, model_name=model_name, new_data=model_info)

            print(model_info)


def main():
    if len(sys.argv) != 2:
        print("Usage: python RunSolver.py BATCH_NUM")
        sys.exit(1)

    global BATCH_NUM
    BATCH_NUM = int(sys.argv[1])
    run_formulation()

if __name__ == "__main__":
    main()
