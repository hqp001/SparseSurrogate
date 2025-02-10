import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from Trainer.Dataset import FashionMNISTDataset

TRAIN_EPOCHS = 5

def train(
    data_seed,
    training_seed,
    n_pixel_1d,
    layer_size,
    n_layers,
    training_epochs,
):
    # Set random seed for reproducibility and select the image that is going to be perturbed
    torch.manual_seed(training_seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training with: ", DEVICE)
    # Get MNIST digit recognition data set
    train_dataloader = FashionMNISTDataset(train=True, n_size_1d = n_pixel_1d, batch_size=32).get_data()
    test_dataloader = FashionMNISTDataset(train=False, n_size_1d = n_pixel_1d, batch_size=64).get_data()

    # Create the neural network
    layers = [nn.Flatten(), nn.Linear(n_pixel_1d**2, layer_size), nn.ReLU()]

    nn.init.xavier_uniform_(layers[1].weight)
    for i in range(n_layers - 1):
        layers.append(nn.Linear(layer_size, layer_size))
        nn.init.xavier_uniform_(layers[-1].weight)
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_size, 10))
    nn.init.xavier_uniform_(layers[-1].weight)
    reg = nn.Sequential(*layers)
    nn.utils.clip_grad_norm_(reg.parameters(), 0.1)


    reg = reg.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(reg.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    for epoch in range(training_epochs):
        reg.train()
        for batch_X, batch_y in train_dataloader:
            # Forward pass
            x = batch_X.to(DEVICE)
            y = batch_y.to(DEVICE)
            outputs = reg(x)

            # Calculate loss
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reg.parameters(), 0.1)
            optimizer.step()


        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")



        # Print training loss after each epoch
        print(f"Epoch [{epoch + 1}/{training_epochs}], Training Loss: {loss.item():.4f}")

    # Test the trained model on the test dataset
    reg.eval()
    correct = 0
    total = 0

    # Determine the accuracy of the trained model on the test dataset
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            x = batch_X.to(DEVICE)
            y = batch_y.to(DEVICE)

            outputs = reg(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    reg = reg.to("cpu")

    return reg, accuracy
