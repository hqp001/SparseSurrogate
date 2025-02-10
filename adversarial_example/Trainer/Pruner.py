import torch
import torch.nn.utils.prune as prune
import numpy as np
import tqdm


class Pruner:

    def __init__(self, sparsity, structured, prune_type):

        self.sparsity = sparsity
        self.structured = structured
        self.prune_type = prune_type

    def prune(self, model):


        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        if self.prune_type == "L1":

            if self.structured:
                for module, param in parameters_to_prune:
                    if isinstance(module, torch.nn.Conv2d):
                        prune.ln_structured(module, name= param, amount= self.sparsity, n=1, dim=0)
                    elif isinstance(module, torch.nn.Linear):
                        prune.ln_structured(module, name=param, amount=self.sparsity, n=1, dim=0)
            else:
                #prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.sparsity)
                for module, param in parameters_to_prune:
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name=param, amount= self.sparsity)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name=param, amount=self.sparsity)

        elif self.prune_type == "random":

            if self.structured:
                for module, param in parameters_to_prune:
                    if isinstance(module, torch.nn.Conv2d):
                        prune.random_structured(module, name= param, amount= self.sparsity, dim=0)
                    elif isinstance(module, torch.nn.Linear):
                        prune.random_structured(module, name=param, amount=self.sparsity, dim=0)

            else:
                for module, param in parameters_to_prune:
                    if isinstance(module, torch.nn.Conv2d):
                        prune.random_unstructured(module, name=param, amount= self.sparsity)
                    elif isinstance(module, torch.nn.Linear):
                        prune.random_unstructured(module, name=param, amount=self.sparsity)



        else:
            raise ValueError(f"{self.prune_type} not found")


        return model

    @staticmethod
    def apply_mask(model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, 'weight')

        return model

