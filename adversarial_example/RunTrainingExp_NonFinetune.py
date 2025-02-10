import time
import copy
from SurrogateLIBTrainFashion import train as SurrogateLIBTrain
import torch
import json
from Non_Finetune import prune

import os

import manager


TRAINING_EPOCHS = 5
ROUNDS = 5
PRUNE_EPOCHS = 5
BASE_FOLDER = "fashion_experiments"


def run_training():

    summary_path = os.path.join(BASE_FOLDER, 'summary.json')
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    for input_id_path in summary_data["models"]:

        input_id = input_id_path["input_id"]
        args = input_id_path["input_arguments"]
        args["training_epochs"] = 0

        model, accuracy = SurrogateLIBTrain(**args)
        model_path, output = manager.get_model_info(input_id=input_id, model_name="dense")
        model.load_state_dict(torch.load(model_path))


        for sparsity in [0.3, 0.5, 0.8, 0.9, 0.95]:
            for structured in [True, False]:
                sparse_model = copy.deepcopy(model)

                start_time = time.time()
                sparse_model, test_score = prune(sparse_model, args['n_pixel_1d'], sparsity, n_rounds=ROUNDS, max_epochs=PRUNE_EPOCHS, structured=structured, prune_type="L1")
                count_time = time.time() - start_time

                output_args = {
                    'test_score': test_score,
                    'sparsity': sparsity,
                    'n_rounds': ROUNDS,
                    'prune_epochs': PRUNE_EPOCHS,
                    'norm': "L1",
                    'generate_time': count_time,
                    'structured': structured,
                    'fine_tune': False
                }

                if structured:
                    manager.insert_model(input_id=input_id, model_name=f'L1_structured_nonfinetune_{sparsity}', output_args=output_args, model=sparse_model)

                else:
                    manager.insert_model(input_id=input_id, model_name=f'L1_unstructured_nonfinetune_{sparsity}', output_args=output_args, model=sparse_model)



def main():
    run_training()

if __name__ == "__main__":
    main()
