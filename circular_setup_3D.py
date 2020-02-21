# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import numpy as np
import os
from imageio import imread, imwrite
import matplotlib.pyplot as plt


class CircularScanning3D:

    def __init__(self):

        self.vol_geom = astra.create_vol_geom(32, 64, 16)
        self.angles = np.linspace(0, np.pi/2, 180,False)
        self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 80, 80, self.angles)



    def run(self, data):
        proj_id, proj_data = astra.create_sino3d_gpu(data, self.proj_geom, self.vol_geom)
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id

        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id, 150)


        rec = astra.data3d.get(rec_id)


        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

        return rec, proj_data


if __name__ == '__main__':



    debug = False
    src = "D:\\Datasets\\lamino_attachable\\"
    dest = "D:\\Datasets\\lamino_attachable_circular_90\\input\\"



    for folder in os.listdir(src):
        plane = np.zeros((16, 32, 64))
        for k in range(16):
            i = imread(src + folder + '\\' + 'slice_{:02d}.png'.format(k), pilmode='F')
            plane[k, :, :] = i


        setup = CircularScanning3D()
        out, proj = setup.run(plane)

        if debug:

            plt.figure()
            plt.imshow(out[12,:,:])
            plt.figure()
            plt.imshow(plane[12,:,:])
            plt.show()
            break
        else:
            if not os.path.isdir(dest):
                os.mkdir(dest)
            if not os.path.isdir(dest + folder):
                os.mkdir(dest + folder)
            for k in range(16):
                imwrite(dest+folder+"\\slice_{:02d}.png".format(k), out[k,:,:])

        print(".")
