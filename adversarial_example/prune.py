from Trainer.Trainer import ModelTrainer
from Trainer.Pruner import Pruner
from Trainer.Dataset import MNISTDataset
from Trainer.ModelHelpers import init_weights, count_params
from copy import deepcopy
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def prune(nn_model, n_size_1d, sparsity, n_rounds, max_epochs):

    ROUNDS = n_rounds
    MAX_EPOCHS = max_epochs

    print("\n\nStart Pruning")
    print("Device to train: ", DEVICE)

    train_loader = MNISTDataset(train=True, n_size_1d = n_size_1d, batch_size=64).get_data()
    test_loader = MNISTDataset(train=False, n_size_1d = n_size_1d, batch_size=64).get_data()

    nn_model = nn_model.to(device=DEVICE)

    total_params = count_params(nn_model)

    trainer = ModelTrainer(max_epochs=MAX_EPOCHS, learning_rate=1e-2, device=DEVICE)

    print("Accuracy before training: ", trainer.calculate_score(nn_model, test_loader))
    print("Total params: ", total_params)

    initial_weights = deepcopy(nn_model.state_dict())
    prune_pc_per_round = 1 - (1 - sparsity) ** (1 / ROUNDS)


    for round in range(ROUNDS):
        print(f"\nPruning round {round} of {ROUNDS}")

        # Fit the model to the training data
        trainer.train(nn_model, train_loader)

        pruned_model = Pruner(sparsity=prune_pc_per_round, structured=True).prune(nn_model)

        # Reset model
        init_weights(nn_model, initial_weights)

        # print(f"Model accuracy: {accuracy:.3f}%")
        # print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    trainer.train(nn_model, train_loader)

    Pruner.apply_mask(nn_model)


    total_params = count_params(nn_model)
    print("Total params after pruning: ", total_params, "\n")

    test_score = trainer.calculate_score(nn_model, test_loader)

    return nn_model, test_score

