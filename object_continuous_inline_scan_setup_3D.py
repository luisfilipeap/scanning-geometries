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
from matplotlib import pyplot as plt



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

    def __init__(self, alpha_param, n_cells_param, n_proj_param, rec_size_param, vert_shift=0, tg_dir="left"):
        """
        It creates a new instance of the class ScanningExecution.
        :param alpha_param: fan-beam opening angle of the X-ray source used in the inline CT setup;
        :param n_cells_param: number of detector elements used in the inline CT setup;
        :param n_proj_param: number of X-ray projections aquired during the object movement;
        :param rec_size_param: number W of pixels of the W x W reconstruction grid;
        :param omega_rotation: total object rotation (in degrees) around its own axis between the first and the last projection
        acquisition.
        """
        self.S = 1
        self.cells = n_cells_param
        self.desired_projs = n_proj_param
        self.vol_geom = astra.create_vol_geom(rec_size_param[0], rec_size_param[1], rec_size_param[2])
        self.setup = InlineScanningSetup3D(alpha=alpha_param, detector_cells=n_cells_param, number_of_projections=self.desired_projs*self.S, object_size=rec_size_param, vert_shift=vert_shift, tg_dir=tg_dir)
        self.proj_geom = astra.create_proj_geom('cone_vec', n_cells_param, n_cells_param, self.setup.get_geometry_matrix())

    def run(self, phantom_param):
        """
        It executes an image reconstruction using the projections acquired in the inline setup.
        :param phantom_param: 2D image of the phantom that should be used to simulate the acquisition of projections from
        real object;
        :param rec_algorithm_param: reconstruction algorithm to be used. The option available are: SIRT_CUDA and FBP_CUDA;
        :param n_iterations_param: number of iterations to be used in case of iterative reconstructions;
        :return: a dictionary containing the reconstructed image into 'rec' index, the reconstruction time into 'time' index,
        and the acquired sinogram into the 'sino' index;
        """

        id_old, proj_data = astra.create_sino3d_gpu(phantom_param, self.proj_geom, self.vol_geom)
        new_proj = np.zeros((self.cells, self.desired_projs, self.cells))

        for j in range(self.desired_projs):
            temp = np.sum(proj_data[:, j*self.S : j*self.S+self.S, :], axis=1)
            new_proj[:,j,:] = temp

        past_geom_matrix = self.setup.get_geometry_matrix()
        self.new_geom_matrix = past_geom_matrix[range(int(self.S/2),past_geom_matrix.shape[0],int(self.S)), :]

        new_geom = astra.create_proj_geom('cone_vec', self.cells, self.cells, self.new_geom_matrix)
        proj_id = astra.data3d.create('-sino', new_geom, new_proj)

        rec_id = astra.data3d.create('-vol', self.vol_geom)
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        start_time = time.time()
        astra.algorithm.run(alg_id, 700)
        elapsed_time = time.time() - start_time


        output = {'rec': astra.data3d.get(rec_id), 'time': elapsed_time, 'sino': new_proj}

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        #astra.data3d.delete(proj_id)
        astra.projector.delete(id_old)

        return output



if __name__ == '__main__':


    p = 200
    a = 50

    save = "sino"
    src = "C:\\Users\\Luis\\Desktop\\PC Antuerpia\\Datasets\\Lamino\\pssp_cfk91_CT1_binning2x2\\SlicesY\\"
    dest = "C:\\Users\\Luis\\Desktop\\PC Antuerpia\\Datasets\\Lamino\\pssp_cfk91_CT1_binning2x2\\lamino-simulation\\"

    if not os.path.isdir(dest):
        os.mkdir(dest)

    data = os.listdir(src)
    plane = np.zeros((187, 565, 547))
    k = 0
    for file in data:
        print(file)
        i = imread(src + file)
        plane[:, :, k] = np.transpose(i) # i = 565x187
        k = k + 1

    np.save('lamino_phantom.npy', plane)

    setup = InlineContinuousScanningObject3D(alpha_param=a, n_cells_param=1200, n_proj_param=p, rec_size_param=(565, 547, 187))
    out = setup.run(plane)


    for k in range(547):
        if not os.path.isdir(dest + "\\rec-continuous\\"):
            os.mkdir(dest + "\\rec-continuous\\")
        final = out['rec']
        imwrite(dest+"\\rec\\slice{:05d}.png".format(k), np.transpose(final[:,:,k]))


    if not os.path.isdir(dest+"\\sino\\"):
        os.mkdir(dest+"\\sino\\")
    for k in range(p):
        imwrite(dest+"\\sino\\proj_{:5d}.png".format(k),out['sino'][:,k,:])



    #plt.figure()
    #plt.imshow(out['sino'][:,2,:], cmap='gray')

    #plt.figure()
    #plt.imshow(out['sino'][:, 6, :], cmap='gray')

    #plt.figure()
    #plt.imshow(out['sino'][:, 9, :], cmap='gray')

    #plt.show()

    #plt.axis('off')
    #plt.ioff()
    #plt.show()