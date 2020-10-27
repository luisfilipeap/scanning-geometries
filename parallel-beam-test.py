import astra
import numpy as np
from imageio import imread, imwrite
import os

vol_geom = astra.create_vol_geom(187, 547, 565)

src = "C:\\Users\\Luis\\Desktop\\PC Antuerpia\\Datasets\\Lamino\\pssp_cfk91_CT1_binning2x2\\SlicesY\\"
angles = np.linspace(0, np.pi, 300,False)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 800, 800, angles, 1200, 95)

data = os.listdir(src)
plane = np.zeros((565, 187, 547))
k = 0
for file in data:
    print(file)
    i = imread(src + file)
    plane[:, :, k] = (i)
    k = k + 1

# Create projection data from this
proj_id, proj_data = astra.create_sino3d_gpu(plane, proj_geom, vol_geom)

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 600)

# Get the result
rec = astra.data3d.get(rec_id)


if not os.path.isdir(".\\REC\\"):
    os.mkdir(".\\REC\\")

if not os.path.isdir(".\\PROJS\\"):
    os.mkdir(".\\PROJS\\")

#for z in range(360):
#    imwrite(".\\PROJS\\proj{:05d}.png".format(z), proj_data[:, z, :])
for k in range(547):
    imwrite(".\\REC\\slice{:05d}.png".format(k), np.transpose(rec[:,:,k]))

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)