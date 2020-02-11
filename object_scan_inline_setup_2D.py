# title                 :scanning_object.py
# description           :This code obtains CT image reconstructions of phantoms scanned in the setup defined by
# 				        instances of the InlineScanningSetup class
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-01-28
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.13.3
# astra_version         :1.8.3
# scipy_version         :1.2.0
# matplotlib_version    :3.0.2
# pilow_version         :5.4.1
# ==============================================================================

from imageio import imread, imwrite
from skimage.transform import resize
from inline_setup_2D import InlineScanningSetup2D
import astra
import time
from scipy import misc
from matplotlib import pyplot as plt


class InlineScanningObject:
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

    def __init__(self, alpha_param, n_cells_param, n_proj_param, rec_size_param=128, omega_rotation=0):
        """
        It creates a new instance of the class ScanningExecution.
        :param alpha_param: fan-beam opening angle of the X-ray source used in the inline CT setup;
        :param n_cells_param: number of detector elements used in the inline CT setup;
        :param n_proj_param: number of X-ray projections aquired during the object movement;
        :param rec_size_param: number W of pixels of the W x W reconstruction grid;
        :param omega_rotation: total object rotation (in degrees) around its own axis between the first and the last projection
        acquisition.
        """

        self.vol_geom = astra.create_vol_geom(rec_size_param, rec_size_param)

        self.setup = InlineScanningSetup2D(alpha=alpha_param, detector_cells=n_cells_param,
                                         number_of_projections=n_proj_param, object_size=rec_size_param,
                                         omega_total=omega_rotation)

        self.proj_geom = astra.create_proj_geom('fanflat_vec', n_cells_param, self.setup.get_geometry_matrix())

    def run(self, phantom_param, rec_algorithm_param='SIRT_CUDA', n_iterations_param=100):
        """
        It executes an image reconstruction using the projections acquired in the inline setup.
        :param phantom_param: 2D image of the phantom that should be used to simulate the acquisition of projections from
        real object;
        :param rec_algorithm_param: reconstruction algorithm to be used. The option available are: SIRT_CUDA and FBP_CUDA;
        :param n_iterations_param: number of iterations to be used in case of iterative reconstructions;
        :return: a dictionary containing the reconstructed image into 'rec' index, the reconstruction time into 'time' index,
        and the acquired sinogram into the 'sino' index;
        """


        proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino(phantom_param, proj_id)

        rec_id = astra.data2d.create('-vol', self.vol_geom)
        cfg = astra.astra_dict(rec_algorithm_param)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)

        if rec_algorithm_param == 'SIRT_CUDA':
            start_time = time.time()
            astra.algorithm.run(alg_id, n_iterations_param)
            elapsed_time = time.time() - start_time
        else:
            astra.algorithm.run(alg_id)
            start_time = time.time()
            elapsed_time = time.time() - start_time

        output = {'rec': astra.data2d.get(rec_id), 'time': elapsed_time, 'sino': sinogram}

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.projector.delete(proj_id)

        return output


if __name__ == '__main__':


    p = 120
    a = 45

    plane = imread("test.tif")
    plane = resize(plane, (128, 128))
    setup = InlineScanningObject(alpha_param=a, n_cells_param=519, n_proj_param=p, rec_size_param=128, omega_rotation=0)
    out = setup.run(plane)

    imwrite("linear_{}_projs_{}_fanbeam.png".format(p, a), out['rec'])

    #plt.figure()
    #plt.imshow(out['rec'], cmap='gray')

    #plt.axis('off')
    #plt.ioff()
    #plt.show()