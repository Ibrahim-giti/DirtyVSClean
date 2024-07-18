""" file for reading the data"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class PlateDataHandler(Dataset):
    def __init__(self, root_dir, img_size=(180, 180), testing = False):
        self.img_size = img_size
        self.data = []
        self.labels = []
        if not testing:
            for label in ['cleaned', 'dirty']:
                path = os.path.join(root_dir, label)
                class_num = 0 if label == 'cleaned' else 1
                for img_name in os.listdir(path):
                    try:
                        img_path = os.path.join(path, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            img = cv2.resize(img, self.img_size)
                            self.data.append(img)
                            self.labels.append(class_num)
                    except Exception as e:
                        print(f"Error loading image {img_name}: {e}")
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)

        else:
            path = root_dir
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        self.data.append(img)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = ToTensor()(img)
        return img, label
    

if __name__ == "__main__":
    p = PlateDataHandler(root_dir='./plates/plates/train')
