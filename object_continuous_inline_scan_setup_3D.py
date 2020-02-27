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
from inline_setup_3D import InlineScanningSetup3D
import astra
import time
import numpy as np
import os
from scipy.io import savemat



class InlineContinuousScanningObject3D:
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

    def __init__(self, alpha_param, n_cells_param, n_proj_param, rec_size_param):
        """
        It creates a new instance of the class ScanningExecution.
        :param alpha_param: fan-beam opening angle of the X-ray source used in the inline CT setup;
        :param n_cells_param: number of detector elements used in the inline CT setup;
        :param n_proj_param: number of X-ray projections aquired during the object movement;
        :param rec_size_param: number W of pixels of the W x W reconstruction grid;
        :param omega_rotation: total object rotation (in degrees) around its own axis between the first and the last projection
        acquisition.
        """
        self.S = 10
        self.cells = n_cells_param
        self.desired_projs = n_proj_param
        self.vol_geom = astra.create_vol_geom(rec_size_param[0], rec_size_param[1], rec_size_param[2])
        self.setup = InlineScanningSetup3D(alpha=alpha_param, detector_cells=n_cells_param, number_of_projections=self.desired_projs*self.S, object_size=rec_size_param)
        self.proj_geom = astra.create_proj_geom('cone_vec', n_cells_param, n_cells_param, self.setup.get_geometry_matrix())

    def run(self, phantom_param, rec_algorithm_param='SIRT3D_CUDA', n_iterations_param=100):
        """
        It executes an image reconstruction using the projections acquired in the inline setup.
        :param phantom_param: 2D image of the phantom that should be used to simulate the acquisition of projections from
        real object;
        :param rec_algorithm_param: reconstruction algorithm to be used. The option available are: SIRT_CUDA and FBP_CUDA;
        :param n_iterations_param: number of iterations to be used in case of iterative reconstructions;
        :return: a dictionary containing the reconstructed image into 'rec' index, the reconstruction time into 'time' index,
        and the acquired sinogram into the 'sino' index;
        """

        _, proj_data = astra.create_sino3d_gpu(phantom_param, self.proj_geom, self.vol_geom)

        new_proj = np.zeros((self.cells, self.desired_projs, self.cells))
        for j in range(self.desired_projs):
            xv, yv = np.meshgrid(range(self.cells), range(self.cells))
            d = np.sqrt(np.power(xv - self.cells / 2, 2) + np.power(yv - self.cells / 2, 2))
            sigma = 10
            m =  1 - np.exp(-(np.power(d,2) / ( 5.0 * sigma**2)))

            #plt.figure()
            #plt.imshow(m)
            #plt.show()

            temp = np.sum(proj_data[:, j*self.S : j*self.S+self.S, :], axis=1)
            #temp = np.ones((self.cells, self.cells))
            #new_proj[:, j, :] = self.my_add_noise_to_sino(sinogram_in=temp, I0=pow(10,3), mask=m)
            new_proj[:,j,:] = temp


        past_geom_matrix = self.setup.get_geometry_matrix()
        self.new_geom_matrix = past_geom_matrix[range(int(self.S/2),past_geom_matrix.shape[0],int(self.S)), :]
        new_geom = astra.create_proj_geom('cone_vec', self.cells, self.cells, self.new_geom_matrix)

        proj_id = astra.data3d.create('-sino', new_geom, new_proj)


        rec_id = astra.data3d.create('-vol', self.vol_geom)
        cfg = astra.astra_dict(rec_algorithm_param)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)

        if rec_algorithm_param == 'SIRT3D_CUDA':
            start_time = time.time()
            astra.algorithm.run(alg_id, n_iterations_param)
            elapsed_time = time.time() - start_time
        else:
            astra.algorithm.run(alg_id)
            start_time = time.time()
            elapsed_time = time.time() - start_time

        output = {'rec': astra.data3d.get(rec_id), 'time': elapsed_time, 'sino': new_proj}

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)

        astra.data3d.delete(proj_id)
        astra.projector.delete(proj_id)

        return output

    def my_add_noise_to_sino(self, sinogram_in, I0, mask, seed=None):
        """Adds Poisson noise to a sinogram.

        :param sinogram_in: Sinogram to add noise to.
        :type sinogram_in: :class:`numpy.ndarray`
        :param I0: Background intensity. Lower values lead to higher noise.
        :type I0: :class:`float`
        :returns:  :class:`numpy.ndarray` -- the sinogram with added noise.

        """

        if not seed == None:
            curstate = np.random.get_state()
            np.random.seed(seed)

        if isinstance(sinogram_in, np.ndarray):
            sinogramRaw = sinogram_in
        else:
            sinogramRaw = astra.data2d.get(sinogram_in)

        max_sinogramRaw = sinogramRaw.max()
        sinogramRawScaled = sinogramRaw / max_sinogramRaw
        # to detector count
        sinogramCT = I0 * np.exp(-sinogramRawScaled)
        # add poison noise
        sinogramCT_C = np.zeros_like(sinogramCT)
        for i in range(sinogramCT_C.shape[0]):
            for j in range(sinogramCT_C.shape[1]):
                diff = sinogramCT[i, j] - np.random.poisson(sinogramCT[i, j])
                if len(mask) > 0:
                    sinogramCT_C[i, j] = sinogramCT[i, j]+diff*mask[i,j]
                else:
                    sinogramCT_C[i, j] = sinogramCT[i, j]+diff
        # to density
        #sinogramCT_C = sinogramCT_C
        sinogramCT_D = sinogramCT_C / I0
        sinogram_out = -max_sinogramRaw * np.log(sinogramCT_D)


        if not isinstance(sinogram_in, np.ndarray):
            astra.data2d.store(sinogram_in, sinogram_out)

        if not seed == None:
            np.random.set_state(curstate)

        return sinogram_out


if __name__ == '__main__':


    p = 10
    a = 30

    save = "setup"
    src = "D:\\Datasets\\lamino_attachable\\"
    dest = "D:\\Datasets\\lamino_attachable_10_projs\\input\\"

    if not os.path.isdir(dest):
        os.mkdir(dest)

    for folder in os.listdir(src):
        plane = np.zeros((16, 32, 64))
        for k in range(16):
            i = imread(src + folder + '\\' + 'slice_{:02d}.png'.format(k), pilmode='F')
            plane[k, :, :] = i


        setup = InlineContinuousScanningObject3D(alpha_param=a, n_cells_param=80, n_proj_param=p, rec_size_param=(32,64,16))
        out = setup.run(plane)

        if save == "setup":

            savemat(dest+"setup.mat", {"cells": setup.cells, "projs": setup.desired_projs, "blur_geom_matrix": setup.new_geom_matrix, "disc_geom_matrix": setup.setup.geometry_matrix})

            break
        elif save == "rec":
            if not os.path.isdir(dest + folder):
                os.mkdir(dest + folder)

            for k in range(16):
                imwrite(dest+folder+"\\slice_{:02d}.png".format(k), out['rec'][k,:,:])
        else:
            savemat(dest+"{}.mat".format(folder), {"projs": out['sino']})



    #plt.figure()
    #plt.imshow(out['sino'][4,:,:], cmap='gray')
    #plt.show()

    #plt.axis('off')
    #plt.ioff()
    #plt.show()