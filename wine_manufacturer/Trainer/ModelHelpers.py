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

def double_transform(model: torch.nn.Module):

    traced = torch.fx.symbolic_trace(model)

    for node in traced.graph.nodes:

        if node.op == 'call_module':

            try:
                layer = getattr(traced, node.target)
            except AttributeError:
                print(f"There is no attribute {node.target}")

            if isinstance(layer, torch.nn.Linear):

                in_features = layer.in_features
                out_features = layer.out_features

                if in_features == 256:
                    in_features = 512
                if out_features == 256:
                    out_features = 512
                new_linear_layer = torch.nn.Linear(in_features, out_features)

                # Copy existing weights to the new layer
                new_linear_layer.weight.data[:layer.out_features, :layer.in_features] = layer.weight.data
                new_linear_layer.bias.data[:layer.out_features] = layer.bias.data

                # Randomly initialize the rest of the weights and biases
                torch.nn.init.kaiming_uniform_(new_linear_layer.weight[layer.out_features:, layer.in_features:], a=math.sqrt(5))
                torch.nn.init.uniform_(new_linear_layer.bias[layer.out_features:])

                with traced.graph.inserting_before(node):
                    traced.add_submodule(f"{node.name}_double", new_linear_layer)
                    new_node = traced.graph.call_module(f"{node.name}_double", node.args, node.kwargs)
                    node.replace_all_uses_with(new_node)

                # Remove the old node from the graph
                traced.graph.erase_node(node)

    traced.recompile()
    traced.delete_all_unused_submodules()

    return traced
