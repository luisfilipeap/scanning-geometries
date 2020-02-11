import astra
from semi_circ_conveyor_belt_2D import SemiCircularConveyorBelt
import time
from imageio import imread, imwrite
from skimage.transform import resize
from matplotlib import pyplot  as plt
import numpy as np

class CircularScanningObject:
    """
    This class defines an inline scanning geometry and executes image reconstructions.
    Attributes
    ----------
    proj_geom   : dict
        It holds the projection geometry to be used;
    setup       : ndarray
        It holds the scanning geometry to be used;
    vol_geom    : dict
        It holds the characteristics of the reconstruction volume;
    Methods
    -------
    run(phantom_param, rec_algorithm_param='SIRT_CUDA', n_iterations_param=100)
        It executes an image reconstruction using the projections acquired in the inline setup.
    """

    def __init__(self, n_projs_param, src_dist_param, det_dist_param, fan_beam_param, radius_param, rec_size_param):


        self.vol_geom = astra.create_vol_geom(rec_size_param, rec_size_param)

        self.setup = SemiCircularConveyorBelt(radius=radius_param, n_projs=n_projs_param, src_dist=src_dist_param, det_dist= det_dist_param, fan_beam_angle=fan_beam_param)
        self.proj_geom = astra.create_proj_geom('fanflat_vec', self.setup.get_det_size(), self.setup.get_geometry_matrix())

    def run(self, phantom_param, rec_algorithm_param='SIRT_CUDA', n_iterations_param=100):



        proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino(phantom_param, proj_id)

        rec_id = astra.data2d.create('-vol', self.vol_geom)

        if rec_algorithm_param == 'SIRT_CUDA':

            cfg = astra.astra_dict(rec_algorithm_param)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id
            alg_id = astra.algorithm.create(cfg)

            start_time = time.time()
            astra.algorithm.run(alg_id, n_iterations_param)
            elapsed_time = time.time() - start_time

            astra.algorithm.delete(alg_id)

            output = {'rec': astra.data2d.get(rec_id), 'time': elapsed_time, 'sino': sinogram}


        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)


        return output


if __name__ == '__main__':


    p = 120
    a = 45

    plane = imread("test.tif")
    plane = resize(plane, (128, 128))

    setup = CircularScanningObject(n_projs_param=p, src_dist_param=200, det_dist_param=100, fan_beam_param=a, radius_param=250, rec_size_param=128)
    #print(setup.setup.get_det_size())
    out = setup.run(plane, rec_algorithm_param = "SIRT_CUDA")


    imwrite("semi_circ_{}_projs_{}_fanbeam.png".format(p,a), out['rec'])
    #final = out['rec']
    #plt.figure("REC")
    #plt.imshow(final, cmap="gray")
    #print(np.min(final))
    #print(np.max(final))


    #plt.figure("SINO")
    #plt.imshow(out['sino'], cmap="gray")

    #plt.show()

