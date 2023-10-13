import os
import glob
from PIL import Image

TRAIN_PATH = './imgs/train_all'
TEST_PATH = './imgs/test'
NEW_TRAIN_PATH = './imgs/train_compressed'
NEW_TEST_PATH = './imgs/test_compressed'
NEW_SIZE = 256

def main(ori_path, new_path, new_size=256):
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    files = glob.glob(f'{ori_path}/*')
    num_files = len(files)

    i = 0
    for f in files:
        if i % 500 == 0:
            print(f'Current: {i+1}/{num_files}')

        title, ext = os.path.splitext(f)
        if ext in ['.jpg', '.png']:
            img = Image.open(f)
            width, height = img.width, img.height
            new_width, new_height = 0, 0
            if width > height:
                new_height = new_size
                new_width = new_size * width // height
            else:
                new_width = new_size
                new_height = new_size * height // width
            img_resize = img.resize((new_width, new_height))
            img_resize.save(title.replace(ori_path, new_path) + ext)
        i += 1

if __name__ == '__main__':
    main(ori_path=TRAIN_PATH, new_path=NEW_TRAIN_PATH, new_size=256)