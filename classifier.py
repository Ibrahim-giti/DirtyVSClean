"""I think I'm going to try a simple CNN

Input: Nx180x180x3 

Output: 0 or 1  (binary classification)
0 = clean
1 = dirty

"""
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class PlateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(), # Flatten the 3D tensor to 1D vector
            nn.Linear(64 * 21 * 21, 128),  # Adjusting the size based on max pooling
            nn.ReLU(),
            nn.Linear(128, 1),  #output layer with 1 value for the sigmoid activation
            nn.Sigmoid()  #sigmoid activation for binary output (0 or 1)
        )  

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    clf = PlateClassifier()
    print(clf)




