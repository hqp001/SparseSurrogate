import torch
import torch.nn.utils.prune as prune
import numpy as np
import tqdm


class Pruner:

    def __init__(self, sparsity = 0):

        self.sparsity = sparsity

    def prune(self, model):


        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.sparsity)


        return model

    @staticmethod
    def apply_mask(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, 'weight')

        return model

