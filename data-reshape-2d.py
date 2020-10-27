import os
from imageio import imread, imwrite
from skimage.transform import rescale
import numpy as np
from scipy.ndimage import zoom

src = "D:\\Datasets\\RealLaminoProjsProcessed\\"
dest = "D:\\Datasets\\RealLaminoProjsProcessed-Light\\"


for file in os.listdir(src):
    img = imread(src+file)
    imwrite(dest+file, rescale(img,0.25))