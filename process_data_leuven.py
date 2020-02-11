
import os
import random
from imageio import imread, imwrite
from skimage.transform import resize

defect = "D:\\Datasets\\kuLeuven\\defect\\"
healthy = "D:\\Datasets\\kuLeuven\\healthy\\"

dest = "D:\\Datasets\\kuLeuven\\sample-128\\"

sampling = 100

for data in [defect, healthy]:
    m = 0
    for folder in os.listdir(data):
        slices = os.listdir(data+folder+"\\stacked0\\merge\\")
        slices = [x for x in slices if x.endswith(".tif")]
        slices = slices[75:250]
        random.shuffle(slices)
        selected = slices[0:sampling]

        s = 0
        for img in selected:
            i = imread(data+folder+"\\stacked0\\merge\\"+img)
            i = resize(i, (128, 128))
            if data == healthy:
                imwrite(dest+"health_sample_{}_slice_{}.png".format(m,s), i)
            else:
                imwrite(dest +"defect_sample_{}_slice_{}.png".format(m, s), i)
            s = s + 1
        m = m + 1


