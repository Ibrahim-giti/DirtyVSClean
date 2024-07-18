# """I think I'm going to try a simple CNN

# Input: Nx180x180x3 

# Output: 0 or 1  (binary classification)
# 0 = clean
# 1 = dirty

# """
# import torch
# from torch import nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor

# class PlateClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, (3, 3)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(32, 64, (3, 3)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(64, 64, (3, 3)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Flatten(), # Flatten the 3D tensor to 1D vector
#             nn.Linear(64 * 21 * 21, 128),  # Adjusting the size based on max pooling
#             nn.ReLU(),
#             nn.Linear(128, 1),  #output layer with 1 value for the sigmoid activation
#             nn.Sigmoid()  #sigmoid activation for binary output (0 or 1)
#         )  

#     def forward(self, x):
#         return self.model(x)

# if __name__ == "__main__":
#     clf = PlateClassifier()
#     print(clf)
import torch
from torch import nn

class PlateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # Input channels: 3 (RGB), Output channels: 32, Kernel size: 3x3
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer with kernel size 2x2 and stride 2
        self.conv2 = nn.Conv2d(32, 64, 3)  # Input channels: 32, Output channels: 64, Kernel size: 3x3
        self.conv3 = nn.Conv2d(64, 64, 3)  # Input channels: 64, Output channels: 64, Kernel size: 3x3
        self.flatten = nn.Flatten()  # Flatten layer to convert 2D feature maps to 1D feature vector

        # Calculate the flattened size after conv layers
        self._to_linear = None
        self.convnet = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            nn.ReLU(),
            self.pool
        )

        self._initialize_flattened_size((3, 180, 180))  # Call the method to determine the flattened size

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)  # Input size: self._to_linear, Output size: 128
        self.fc2 = nn.Linear(128, 1)  # Input size: 128, Output size: 1 (binary classification)

    def _initialize_flattened_size(self, input_shape):
        # Dummy forward pass to calculate flattened size
        x = torch.rand(1, *input_shape)  # Create a dummy tensor with the input shape
        x = self.convnet(x)  # Pass it through the convolutional network
        self._to_linear = x.numel()  # Calculate the number of elements in the output tensor

    def forward(self, x):
        x = self.convnet(x)  # Pass the input through the convolutional network
        x = self.flatten(x)  # Flatten the output
        x = self.fc1(x)  # Pass through the first fully connected layer
        x = self.fc2(x)  # Pass through the second fully connected layer
        return x  # Return the final output

if __name__ == "__main__":
    clf = PlateClassifier()
    print(clf)

