import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
def train(
    data_seed=42,
    training_seed=42,
    n_pixel_1d=16,
    layer_size=16,
    n_layers=2,
    test=True,
    formulation="sos",
    build_only=False,
):
    # Set random seed for reproducibility and select the image that is going to be perturbed
    data_random_state = np.random.RandomState(data_seed)
    image_number = data_random_state.randint(low=0, high=30000)
    torch.manual_seed(training_seed)

    # Define transformations for the MNIST dataset
    transform = transforms.Compose(
        [
            transforms.Resize(n_pixel_1d),  # Resize the image
            transforms.ToTensor(),  # Convert images to tensors. This automatically normalises to [0,1]
        ]
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training with: ", DEVICE)
    # Get MNIST digit recognition data set
    train_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=False, transform=transform, download=True
    )
    # Create DataLoader for handling batches
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    optimizer = optim.Adam(reg.parameters(), weight_decay=0.001)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
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
            optimizer.step()

        # Print training loss after each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")

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


    return reg
