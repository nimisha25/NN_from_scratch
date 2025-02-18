import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class SimpleNN(nn.Module) :
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784,128)   # First linear layer, Hidden layer: 128 (power of 2, between 784 and 10).
        self.relu = nn.ReLU()           # Activation function (ReLU)          
        self.fc2 = nn.Linear(128,10)    # Second linear layer (output layer)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))      # First layer + ReLU activation
        out = self.fc2(x)               # Second layer (no activation here)      
        return out
    

if __name__ == '__main__' :
    # #transforms.Normalize((mean,), (std,))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Download and load the training and validation data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()   #combines Log softmax for class probabilities, -ve log-likelihood for computing loss.

    # 4. Train the Model
    num_epochs = 10  # Train for 10 epochs
    updates = 0
    max_updates = 10000  # Stop after 10,000 weight updates
    training_loss = []
    training_error = []
    validation_loss = []
    validation_error = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for batch_idx, (x, y) in enumerate(train_loader):
            # Flatten the images (batch_size, 784)
            x = x.view(x.size(0), -1)

            # Forward pass
            outputs = model(x)      # same as model.forward(x)
            loss = criterion(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            _, predicted = torch.max(outputs, 1)
            error = (predicted != y).float().mean().item()
            training_loss.append(loss.item())
            training_error.append(error)

            updates += 1
            if updates % 100 == 0:
                print(f"Update {updates}, Loss: {loss.item():.4f}, Error: {error:.4f}")
            
            # Validate every 1000 updates
            if updates % 1000 == 0:
                model.eval()  # Set model to evaluation mode
                val_loss = 0.0
                val_error = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.view(x_val.size(0), -1)
                        val_outputs = model(x_val)
                        val_loss += criterion(val_outputs, y_val).item() * len(y_val)
                        _, val_pred = torch.max(val_outputs, 1)
                        val_error += (val_pred != y_val).sum().item()
                
                val_loss /= len(val_dataset)
                val_error /= len(val_dataset)
                validation_loss.append(val_loss)
                validation_error.append(val_error)
                print(f"Validation - Loss: {val_loss:.4f}, Error: {val_error:.4f}")
            
            if updates >= max_updates:
                break
        if updates >= max_updates:
            break
    
    # 5. Plot Metrics
    plt.figure(figsize=(12, 6))

    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(training_loss)), training_loss, label='Training Loss')
    plt.xlabel("Weight Updates")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Weight Updates")
    plt.legend()

    # Training Error
    plt.subplot(2, 2, 2)
    plt.plot(range(len(training_error)), training_error, label='Training Error', color='orange')
    plt.xlabel("Weight Updates")
    plt.ylabel("Error")
    plt.title("Training Error vs. Weight Updates")
    plt.legend()

    # Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss', color='green')
    plt.xlabel("Weight Updates (x1000)")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs. Weight Updates")
    plt.legend()

    # Validation Error
    plt.subplot(2, 2, 4)
    plt.plot(range(len(validation_error)), validation_error, label='Validation Error', color='red')
    plt.xlabel("Weight Updates (x1000)")
    plt.ylabel("Error")
    plt.title("Validation Error vs. Weight Updates")
    plt.legend()

    plt.tight_layout()
    plt.show()

