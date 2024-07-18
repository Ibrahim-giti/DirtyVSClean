""" The training loop for the classifier"""

import torch
from torch import nn, optim
from read_data import PlateDataHandler
from classifier import PlateClassifier
from torch.utils.data import DataLoader

# Hyper parameters
batch_size = 32
learning_rate= 0.0001
num_epochs = 10

train_dataset = PlateDataHandler(root_dir='./plates/plates/train')
test_dataset = PlateDataHandler(root_dir='./plates/plates/test', testing=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = PlateClassifier()
loss_criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(imgs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%')