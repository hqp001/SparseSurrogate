import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets, transforms
from pyscipopt import Model, quicksum

from pyscipopt_ml.add_predictor import add_predictor_constr

def create_neural_network(model_path):

    layers = [nn.Flatten(), nn.Linear(n_pixel_1d**2, layer_size), nn.ReLU()]
    for i in range(n_layers - 1):
        layers.append(nn.Linear(layer_size, layer_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_size, 10))
    reg = nn.Sequential(*layers)

    reg.load_state_dict(torch.load(model_path, map_location="cpu"))
    reg.eval()

    return reg


def formulate(
    sparsity=0,
    data_seed=42,
    training_seed=42,
    n_pixel_1d=16,
    layer_size=16,
    n_layers=2,
    formulation="sos",
):

    # Create the neural network
    dense_path = f"./models/mnist_{n_pixel_1d}_{n_layers}_{layer_size}_{data_seed}_{training_seed}_dense.pth"

    if sparsity == 0:
        model_path = f"./models/mnist_{n_pixel_1d}_{n_layers}_{layer_size}_{data_seed}_{training_seed}_dense.pth"
    else:
        model_path = f"./models/mnist_{n_pixel_1d}_{n_layers}_{layer_size}_{data_seed}_{training_seed}_{sparsity}.pth"

    dense_model = create_neural_network(dense_path)
    sparse_model = create_neural_network(model_path)

    data_random_state = np.random.RandomState(data_seed)
    image_number = data_random_state.randint(low=0, high=30000)
    torch.manual_seed(training_seed)

    transform = transforms.Compose(
        [
            transforms.Resize(n_pixel_1d),  # Resize the image
            transforms.ToTensor(),  # Convert images to tensors. This automatically normalises to [0,1]
        ]
    )

    # Get MNIST digit recognition data set
    train_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=False, transform=transform, download=True
    )
    # values most change the classification of an image
    scip = Model()

    output_values = sparse_model.forward(train_dataset[image_number][0])
    sorted_labels = torch.argsort(output_values)
    right_label = sorted_labels[0][-1]
    wrong_label = sorted_labels[0][-2]

    # Create the input variables
    input_vars = np.zeros((1, n_pixel_1d, n_pixel_1d), dtype=object)
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            # Tight bounds are important for MIP formulations of neural networks. They often drastically improve
            # performance. As our data is a scaled image, it is in the range [0, 1].
            # These bounds will then propagate to other variables.
            input_vars[0][i][j] = scip.addVar(name=f"x_{i}_{j}", vtype="C", lb=0, ub=1)

    # Create the output variables. (Note that these variables will be automatically constructed if not specified)
    output_vars = np.zeros((10,), dtype=object)
    for i in range(10):
        output_vars[i] = scip.addVar(name=f"y_{i}", vtype="C", lb=None, ub=None)

    # Create the difference variables
    sum_max_diff = data_random_state.uniform(low=4.5, high=5.5)
    abs_diff = np.zeros((n_pixel_1d, n_pixel_1d), dtype=object)
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            abs_diff[i][j] = scip.addVar(name=f"abs_diff_{i}", vtype="C", lb=0, ub=1)

    # Create constraints ensuring only a total certain amount of the picture can change
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            scip.addCons(
                abs_diff[i][j] >= input_vars[0][i][j] - train_dataset[image_number][0][0][i][j]
            )
            scip.addCons(
                abs_diff[i][j] >= -input_vars[0][i][j] + train_dataset[image_number][0][0][i][j]
            )

    scip.addCons(
        quicksum(quicksum(abs_diff[i][j] for j in range(n_pixel_1d)) for i in range(n_pixel_1d))
        <= sum_max_diff
    )

    # Set an objective to maximise the difference between the correct and the wrong label
    scip.setObjective(-output_vars[wrong_label] + output_vars[right_label] + 1)

    # Add the ML constraint
    pred_cons = add_predictor_constr(
        scip,
        sparse_model,
        input_vars,
        output_vars,
        unique_naming_prefix="adversarial_",
        formulation=formulation,
    )

    return dense_model, sparse_model, scip
