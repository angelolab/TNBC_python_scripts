# functions for creating image overlays
import os
import skimage.io as io
import numpy as np
from skimage.measure import block_reduce

from ark.utils import io_utils

# get FOVs of size 2048 x 2048
image_dir = '/Volumes/Big_Boy/TONIC_Cohort/image_data/samples/'
folders = io_utils.list_folders(image_dir)

keep_folders = []

for folder in folders:
    test_img = io.imread(os.path.join(image_dir, folder, 'CD3.tiff'))
    if test_img.shape[0] == 2048:
        keep_folders.append(folder)

output_img = np.zeros((5120, 10240))

# final image size: 20 x 40 images
image_num = 0
channel_name = 'ECAD.tiff'

np.random.shuffle(keep_folders)
for col_num in range(40):
    for row_num in range(20):
        img = io.imread(os.path.join(image_dir, keep_folders[image_num], channel_name))
        img_small = block_reduce(img, block_size=(8, 8), func=np.mean)
        output_img[row_num * 256:(row_num + 1) * 256, col_num * 256:(col_num + 1) * 256] = img_small
        image_num += 1

output_img = output_img / np.max(output_img)
io.imsave('/Users/noahgreenwald/Downloads/ECAD.tiff', output_img)

