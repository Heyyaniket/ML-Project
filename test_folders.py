import os

try:
    normal_count = len(os.listdir(r'data/train/NORMAL'))
    pneumonia_count = len(os.listdir(r'data/train/PNEUMONIA'))
    print(f"SUCCESS! Found {normal_count} Normal images and {pneumonia_count} Pneumonia images.")
except FileNotFoundError:
    print("ERROR: The folders still don't exist. The download failed.")