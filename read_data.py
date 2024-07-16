""" file for reading the data"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PlateDataHandler:
    #apperently forward slashes work on all operating systems and universilly known for file paths
    root_dir = r".\plates\plates\train"  
    dir_test: str = r".\plates\plates\test"  
    img_size: Tuple[int, int] = (180, 180) 

    data: List[np.ndarray] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    img_size: Tuple[int, int] = (180, 180)

    def load_data(self):
        for label in ['cleaned', 'dirty']:
            path = os.path.join(self.root_dir, label)
            class_num = 0 if label == 'cleaned' else 1
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    resized_array = cv2.resize(img_array, self.img_size)
                    self.data.append(resized_array)
                    self.labels.append(class_num)
                except Exception as e:
                    print(f"Error loading image {img}: {e}")


    def get_data(self):
        return np.array(self.data)
    
    def get_labels(self):
        return np.array(self.labels)
if __name__ == "__main__":
    p = PlateDataHandler()
    p.load_data()

    labels =  p.get_labels()

    print(labels)