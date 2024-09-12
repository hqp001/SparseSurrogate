import copy
from SurrogateLIBTrain import train as SurrogateLIBTrain
import torch
from prune import prune

import os

import manager


TRAINING_EPOCHS = 5
ROUNDS = 5
PRUNE_EPOCHS = 5

def generate_arguments():
    for d_s, t_s in [(0, 0), (1, 3), (8, 12), (90, 99)]:
        for n_p in [28, 22, 18]:
            for n_e, d in [(4, 32), (4, 64), (5, 32), (5, 64)]:
                yield {
                    'data_seed': d_s,
                    'training_seed': t_s,
                    'n_pixel_1d': n_p,
                    'n_layers': n_e,
                    'layer_size': d,
                    'training_epochs': TRAINING_EPOCHS
                }

def run_training():

    adversarial_data = []

    for args in generate_arguments():

        model, accuracy = SurrogateLIBTrain(**args)

        input_id = manager.insert_input_arguments(input_args=args)

        output_args = {
            'accuracy': accuracy,
            'sparsity': 0
        }

        #torch.save(model.state_dict(), f"./adversarial_example/models/{model_id}/dense.pth")
        manager.insert_model(input_id=input_id, model_name='dense', output_args=output_args, model=model)

        for sparsity in [0.5, 0.8, 0.9]:
            sparse_model = copy.deepcopy(model)
            sparse_model, test_score = prune(sparse_model, args['n_pixel_1d'], sparsity, n_rounds=ROUNDS, max_epochs=PRUNE_EPOCHS)
            output_args = {
                'dense_accuracy': accuracy,
                'test_score': test_score,
                'sparsity': sparsity,
                'n_rounds': ROUNDS,
                'prune_epochs': PRUNE_EPOCHS,
                'structured': True
            }
            manager.insert_model(input_id=input_id, model_name=f'sparse_{sparsity}', output_args=output_args, model=sparse_model)
            #torch.save(sparse_model.state_dict(), f"./adversarial_example/models/{model_id}/sparse_{sparsity}.pth")

def main():
    run_training()

if __name__ == "__main__":
    main()
