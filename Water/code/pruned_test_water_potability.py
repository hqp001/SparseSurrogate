import numpy as np
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from utils import read_csv_to_dict, train_torch_neural_network
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.utils.prune as prune
import copy
from src.pyscipopt_ml.add_predictor import add_predictor_constr
import torch.optim as optim
import torch.nn as nn

def count_nonzero_parameters(model):
    non_zero_weights = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            non_zero_weights += torch.count_nonzero(module.weight).item()
    return non_zero_weights

def copy_weights_to_new_model(old_model, new_model):
    with torch.no_grad():
        for old_layer, new_layer in zip(old_model, new_model):
            if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                new_layer.weight.data = old_layer.weight.data.clone()
                new_layer.bias.data = old_layer.bias.data.clone()

def prune_and_fine_tune_lottery_ticket(model, X_train, y_train, initial_state_dict, n_rounds, total_sparsity, n_epochs=10, lr=0.01):
    prune_pc_per_round = 1 - (1 - total_sparsity) ** (1 / n_rounds)

    # Determine the correct loss function based on the model output
    if model[-1].__class__.__name__ == 'Sigmoid':
        criterion = nn.BCELoss()
        y_train = y_train.float().view(-1, 1)  # Ensure targets are the correct shape and type for BCELoss
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=0.001)

    for round in range(n_rounds):

        dataloader = DataLoader(
            TensorDataset(torch.Tensor(X_train), y_train),
            batch_size=64,
            shuffle=True,
        )
        for epoch in range(n_epochs):
            for batch_X, batch_y in dataloader:
                model.train()
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Print the number of non-zero parameters after fine-tuning
        non_zero_params = count_nonzero_parameters(model)
        print(f"After fine-tuning round {round + 1}, non-zero parameters: {non_zero_params}")
        
        # Prune the model while preserving zeros
        parameters_to_prune = [(module, 'weight') for module in model if isinstance(module, nn.Linear)]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_pc_per_round)

        # Print the number of non-zero parameters after pruning
        non_zero_params = count_nonzero_parameters(model)
        print(f"After pruning round {round + 1}, non-zero parameters: {non_zero_params}")

    # Final fine-tuning after all pruning rounds
    for epoch in range(n_epochs):
        for batch_X, batch_y in dataloader:
            model.train()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Final count of non-zero parameters
    non_zero_params = count_nonzero_parameters(model)
    print(f"Final non-zero parameters: {non_zero_params}")

    return model




def build_and_optimise_water_potability(
    data_seed=42,
    training_seed=42,
    predictor_type="mlp",
    formulation="sos",
    n_water_samples=50,
    layer_size=16,
    max_depth=5,
    n_estimators_layers=3,
    framework="torch",
    build_only=False,
    total_sparsity=0.8,
    n_pruning_rounds=10,
):
    assert predictor_type in ("mlp", "gbdt")
    # Random seed initialisation
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)
    remove_feature_budgets = (
        data_random_state.uniform(1.8, 2.5),
        data_random_state.uniform(20, 25),
        data_random_state.uniform(180, 250),
        data_random_state.uniform(0.9, 1.1),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(1.2, 1.8),
        data_random_state.uniform(4.5, 6),
        data_random_state.uniform(0.9, 1.1),
    )
    add_feature_budgets = (
        data_random_state.uniform(1.8, 2.5),
        data_random_state.uniform(20, 25),
        data_random_state.uniform(180, 250),
        data_random_state.uniform(0.9, 1.1),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(1.2, 1.8),
        data_random_state.uniform(4.5, 6),
        data_random_state.uniform(0.9, 1.1),
    )

    # Path to water potability data
    data_dict = read_csv_to_dict("./data/water_quality.csv")

    # The features of our predictor. All distance based features are variables.
    features = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]
    n_features = len(features)

    # Generate the actual input data arrays for the ML predictors
    X = []
    y = np.array([int(x) for x in data_dict["Potability"]]).reshape(-1)
    for feature in features:
        X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Get the indices of some water that is currently undrinkable
    undrinkable_water_indices = (y == 0).nonzero()[0]
    np.random.shuffle(undrinkable_water_indices)
    undrinkable_water_indices = undrinkable_water_indices[:n_water_samples]

    # Build the MLP classifier
    if predictor_type == "gbdt":
        clf = GradientBoostingClassifier(
            random_state=training_random_state,
            max_depth=max_depth,
            n_estimators=n_estimators_layers,
        ).fit(X, y)
    else:
        if framework == "sklearn":
            hidden_layers = tuple([layer_size for i in range(n_estimators_layers)])
            clf = MLPClassifier(
                random_state=training_random_state,
                hidden_layer_sizes=hidden_layers,
            ).fit(X, y)
        elif framework == "torch":
            clf = train_torch_neural_network(
                X,
                y,
                n_estimators_layers,
                layer_size,
                training_seed,
                reshape=True,
                binary_classifier=True,
            )

            # Prune and fine-tune the PyTorch model
            initial_state_dict = copy.deepcopy(clf.state_dict())
            clf = prune_and_fine_tune_lottery_ticket(
                clf, 
                X_train=torch.tensor(X, dtype=torch.float32), 
                y_train=torch.tensor(y, dtype=torch.long), 
                initial_state_dict=initial_state_dict, 
                n_rounds=n_pruning_rounds, 
                total_sparsity=total_sparsity, 
                n_epochs=10, 
                lr=0.01
            )
            new_clf = train_torch_neural_network(
                X,
                y,
                n_estimators_layers,
                layer_size,
                training_seed,
                reshape=True,
                binary_classifier=True,
            )
            copy_weights_to_new_model(clf, new_clf)
            clf = new_clf
            

        else:
            raise ValueError(f"Unknown framework: {framework}")

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of each water sample after treatment
    feature_vars = np.zeros((n_water_samples, n_features), dtype=object)
    features_removed = np.zeros((n_water_samples, n_features), dtype=object)
    features_added = np.zeros((n_water_samples, n_features), dtype=object)
    if framework in ["sklearn", "torch"] or predictor_type == "gbdt":
        drinkable_water = np.zeros((n_water_samples, 1), dtype=object)
    else:
        drinkable_water = np.zeros((n_water_samples, 2), dtype=object)

    for i in range(n_water_samples):
        for j in range(drinkable_water.shape[-1]):
            drinkable_water[i][j] = scip.addVar(vtype="B", name=f"drinkable_{i}_{j}")
        for j in range(n_features):
            feature_vars[i][j] = scip.addVar(vtype="C", name=f"feature_val_{i}_{j}")
            features_added[i][j] = scip.addVar(
                vtype="C", lb=0, ub=add_feature_budgets[j], name=f"feature_add_{i}_{j}"
            )
            features_removed[i][j] = scip.addVar(
                vtype="C", lb=0, ub=remove_feature_budgets[j], name=f"feature_rem_{i}_{j}"
            )

    for i in range(n_water_samples):
        for j, feature in enumerate(features):
            lb = X[undrinkable_water_indices[i]][j] - remove_feature_budgets[j]
            ub = X[undrinkable_water_indices[i]][j] + add_feature_budgets[j]
            if lb >= 0:
                scip.chgVarLb(feature_vars[i][j], lb)
            scip.chgVarUb(feature_vars[i][j], ub)
            scip.addCons(
                X[undrinkable_water_indices[i]][j] - features_removed[i][j] + features_added[i][j]
                == feature_vars[i][j],
                name=f"change_{i}_{j}",
            )

    for j, feature in enumerate(features):
        scip.addCons(
            sum(features_added[i][j] for i in range(n_water_samples)) <= add_feature_budgets[j],
            name=f"add_budget_{feature}",
        )
        scip.addCons(
            sum(features_removed[i][j] for i in range(n_water_samples))
            <= remove_feature_budgets[j],
            name=f"remove_budget_{feature}",
        )

    # Add the ML predictor for predicting water quality of the sample
    if framework == "sklearn" or predictor_type == "gbdt":
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
            epsilon=0.0001,
            formulation=formulation,
        )
    else:
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
            output_type="classification",
            formulation=formulation,
        )

    # Set the object to maximise the amount of drinkable water after treatment
    if framework in ["sklearn", "torch"] or predictor_type == "gbdt":
        scip.setObjective(-np.sum(drinkable_water) + n_water_samples)
    else:
        scip.setObjective(-np.sum(drinkable_water[:, 1]) + n_water_samples)

    if not build_only:
        # Optimise the SCIP Model
        scip.optimize()

        # We can check the "error" of the MIP embedding via the difference between SKLearn and SCIP output
        if np.max(pred_cons.get_error()) > 10**-3:
            error = np.max(pred_cons.get_error())
            # TODO: There is currently no way to ensure SCIP numerical tolerances dont incorrectly classify

    return scip


def test_water_potability_sklearn_mlp():
    scip = build_and_optimise_water_potability(
        data_seed=42,
        training_seed=42,
        predictor_type="mlp",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="sklearn",
        build_only=False,
    )


def test_water_potability_sklearn_mlp_bigm():
    scip = build_and_optimise_water_potability(
        data_seed=42,
        training_seed=42,
        predictor_type="mlp",
        formulation="bigm",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="sklearn",
        build_only=False,
    )


def test_water_potability_keras():
    scip = build_and_optimise_water_potability(
        data_seed=20,
        training_seed=20,
        predictor_type="mlp",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="keras",
        build_only=False,
    )


def test_water_potability_keras_bigm():
    scip = build_and_optimise_water_potability(
        data_seed=20,
        training_seed=20,
        predictor_type="mlp",
        formulation="bigm",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="keras",
        build_only=False,
    )


def test_water_potability_gbdt():
    scip = build_and_optimise_water_potability(
        data_seed=18,
        training_seed=18,
        predictor_type="gbdt",
        n_water_samples=50,
        layer_size=16,
        max_depth=4,
        n_estimators_layers=4,
        framework="keras",
        build_only=False,
    )
