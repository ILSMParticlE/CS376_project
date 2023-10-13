import os
import shutil

TRAIN_PATH = './imgs/train'
OUPUT_PATH = './imgs/train_all'

train_folders = os.listdir(TRAIN_PATH)
train_files = []
for foler in train_folders:
    # shutil.copytree(f'{TRAIN_PATH}/{foler}', OUPUT_PATH)
    for img_file in os.listdir(f'{TRAIN_PATH}/{foler}'):
        shutil.copy(f'{TRAIN_PATH}/{foler}/{img_file}', f'{OUPUT_PATH}/{img_file}')