import torch
import math

def count_params(model):
    non_zero_weights = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            non_zero_weights += torch.count_nonzero(module.weight).item()
        if isinstance(module, torch.nn.Conv2d):
            non_zero_weights += torch.count_nonzero(module.weight).item()

    return non_zero_weights

def init_weights(model, initial_weights):

    pruned_state_dict = model.state_dict()

    for parameter_name, parameter_values in initial_weights.items():
        # Pruned weights are called <parameter_name>_orig
        augmented_parameter_name = parameter_name + "_orig"

        if augmented_parameter_name in pruned_state_dict:
            pruned_state_dict[augmented_parameter_name] = parameter_values
        else:
            # Parameter name has not changed
            # e.g. bias or weights from non-pruned layer
            pruned_state_dict[parameter_name] = parameter_values

    model.load_state_dict(pruned_state_dict)

    return model
