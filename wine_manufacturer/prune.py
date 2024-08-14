from Trainer.Trainer import ModelTrainer
from Trainer.Pruner import Pruner
from Trainer.Dataset import MNISTDataset
from ModelHelpers import init_weights, count_params
from copy import deepcopy
import torch

ROUNDS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def prune(nn_model, dataloader, sparsity):


    nn_model = nn_model.to(device=DEVICE)

    total_params = count_params(nn_model)

    trainer = ModelTrainer(max_epochs=1, learning_rate=1e-2, device=DEVICE)

    print("Accuracy before training: ", trainer.calculate_score(nn_model, dataloader))
    print("Total params: ", total_params)

    initial_weights = deepcopy(nn_model.state_dict())
    prune_pc_per_round = 1 - (1 - sparsity) ** (1 / ROUNDS)


    for round in range(ROUNDS):
        print(f"\nPruning round {round} of {ROUNDS}")

        # Fit the model to the training data
        trainer.train(nn_model, dataloader)

        pruned_model = Pruner(sparsity=prune_pc_per_round).prune(nn_model)

        # Reset model
        init_weights(nn_model, initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    trainer = ModelTrainer(max_epochs=1, learning_rate=1e-3, device=DEVICE)
    trainer.train(nn_model, dataloader)

    Pruner.apply_mask(nn_model)


    total_params = count_params(nn_model)
    print("Total params after pruning: ", total_params)

    test_score = trainer.calculate_score(nn_model, dataloader)

    return nn_model

