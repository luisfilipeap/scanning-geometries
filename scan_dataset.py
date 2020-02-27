
import os
from object_scan_inline_setup_2D import InlineScanningObject
from object_scan_semi_circ_2D import CircularScanningObject
from imageio import imread, imwrite
from scipy.io import savemat


save = "sino"
type = "inline"
projs = 7
fanbeam = 60

src = "D:\\Datasets\\kuLeuven\\sample-128\\"
dest = "D:\\Datasets\\kuLeuven\\scan_{}_projs_{}_fanbeam_{}\\".format(type, projs, fanbeam)

if not os.path.exists(dest):
    os.mkdir(dest)




if type == "curve":
    setup = CircularScanningObject(n_projs_param=projs, src_dist_param=200, det_dist_param=100, fan_beam_param=fanbeam,
                                   radius_param=250, rec_size_param=128)
else:
    setup = InlineScanningObject(alpha_param=fanbeam, n_cells_param=519, n_proj_param=projs, rec_size_param=128, omega_rotation=0)

if save == "image":
    if not os.path.exists(dest + "input\\"):
        os.mkdir(dest + "input\\")

    for im_name in os.listdir(src):
        plane = imread(src+im_name)
        out = setup.run(plane, rec_algorithm_param="SIRT_CUDA")
        imwrite(dest+"input\\"+im_name, out['rec'])
elif save == "setup":
    savemat("{}_setup_{}_projs_{}_fanbeam.mat".format(type, projs, fanbeam), {"matrix": setup.setup.geometry_matrix, "det_size": setup.setup.det_size, "rec_size": 128})

elif save == "sino":
    if not os.path.exists(dest + "sino\\"):
        os.mkdir(dest + "sino\\")

    for im_name in os.listdir(src):
        plane = imread(src+im_name)
        out = setup.run(plane, rec_algorithm_param="SIRT_CUDA")
        imwrite(dest+"sino\\"+im_name, out['sino'])
