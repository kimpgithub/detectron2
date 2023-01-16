# main.py

import os
from Binary import *

detector = Detector(model_type="IS")


folder_path = input("Enter the path of the folder containing the images: ")

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    detector.onImage(file_path)
    
# detector.onImage("image/2.png")