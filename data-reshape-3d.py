
import os
from imageio import imread, imwrite
from skimage.transform import rescale
import numpy as np
from scipy.ndimage import zoom

src = "D:\\Datasets\\LaminoPhantomCT-Light\\"
dest = "D:\\Datasets\\LaminoPhantom-Light\\input\\"


for folder in os.listdir(src):
    v_in = np.zeros((32, 64, 64))
    count = 0
    for file in os.listdir(src+folder):
        if file.endswith("png"):
            k = imread(src+folder+"\\"+file)
            v_in[count,:,:] = k
            count = count + 1

    v_out = zoom(v_in, (0.5, 0.5, 0.5))

    Z, _, _ = v_out.shape

    if not os.path.isdir(dest+folder):
        os.mkdir(dest+folder)

    for z in range(Z):
        imwrite(dest+folder+"\\slice{:04d}.png".format(z), v_out[z,:,:])