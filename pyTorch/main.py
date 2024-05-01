import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
# Set the seed for reproducibility
np.random.seed(0)

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
        self.layer1 = nn.Linear(in_features=784, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=10)
        # Initialize weights and biases to 0.1
        # with torch.no_grad():
        #     layer1_weights = torch.from_numpy(np.random.randn(self.layer1.out_features,self.layer1.in_features).astype(np.float32))*0.01
        #     layer1_bias = torch.from_numpy(np.random.randn(self.layer1.out_features).astype(np.float32))*0.01
        #     layer2_weights = torch.from_numpy(np.random.randn(self.layer2.out_features, self.layer2.in_features).astype(np.float32))*0.01
        #     layer2_bias = torch.from_numpy(np.random.randn(self.layer2.out_features).astype(np.float32))*0.01
        #     layer3_weights = torch.from_numpy(np.random.randn(self.layer3.out_features, self.layer3.in_features).astype(np.float32)) * 0.01
        #     layer3_bias = torch.from_numpy(np.random.randn(self.layer3.out_features).astype(np.float32))*0.01
        #     self.layer1.weight.copy_(layer1_weights)
        #     self.layer1.bias.copy_(layer1_bias)
        #     self.layer2.weight.copy_(layer2_weights)
        #     self.layer2.bias.copy_(layer2_bias)
        #     self.layer3.weight.copy_(layer3_weights)
        #     self.layer3.bias.copy_(layer3_bias)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), 1)
        return x

BATCH_SIZE = 4000
TOTAL_SIZE = 60000
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../mnist', train=True, transform=transform)
test_dataset = datasets.MNIST(root='../mnist', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True, collate_fn=custom_collate)
# Set up the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    testCorrect = 0
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
    with torch.no_grad():
        for images, labels, encoding in test_loader:
            images = images.view(images.size(0), -1)
        
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, 1)
            testCorrect += torch.sum(predictions==labels)
    accuracy = total_correct / TOTAL_SIZE
    testAccuracy = testCorrect / 10000
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.8f}, Accuracy: {accuracy*100:10.4f}%\tTest Accuracy: {testAccuracy*100:10.4f}%')

print("Training complete.")