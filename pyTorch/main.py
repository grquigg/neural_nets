import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set the seed for reproducibility
torch.manual_seed(0)

def one_hot_encode(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Custom collate function for DataLoader
def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    one_hot_labels = one_hot_encode(labels)
    return images, labels, one_hot_labels

# Define the Neural Network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)
        # Initialize weights and biases to 0.1
        # nn.init.constant_(self.layer1.weight, 0.1)
        # nn.init.constant_(self.layer1.bias, 0.1)
        # nn.init.constant_(self.layer2.weight, 0.1)
        # nn.init.constant_(self.layer2.bias, 0.1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), 1)
        return x

BATCH_SIZE = 4000
TOTAL_SIZE = 60000
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='C:/Users/grego/Documents/NeuralNetworks/mnist', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

# Set up the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training the model
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for images, labels, one_hot in train_loader:
        # Flatten the images
        images = images.view(images.size(0), -1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, one_hot)
        predictions = torch.argmax(outputs, 1)
        total_correct += torch.sum(predictions==labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    accuracy = total_correct / TOTAL_SIZE
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy*100}%')

print("Training complete.")