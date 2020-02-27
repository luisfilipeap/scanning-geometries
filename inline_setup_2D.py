# title           :inline_scanning_setup.py
# description     :This code simulates an inline Compute Tomography (CT) scanning setup
# author          :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date            :2019-01-28
# version         :1.0
# notes           :Please let me know if you find any problem in this code
# python_version  :3.6
# numpy_version   :1.13.3
# ==============================================================================

from math import floor, radians, tan, atan2, sin, cos
import numpy as np


class InlineScanningSetup2D:
    """
    This class represents a Compute Tomography (CT) inline scanning setup composed of a fixed X-ray source and detector
    for imaging objects passing on a conveyor belt.
    Attributes
    ---------
    geometry_matrix : ndarray
        It holds the position of each element (object, X-ray source, and detector) in a vector space at each projection
        acquisition. The data is encapsulated according to the specifications of ASTRA Toolbox.
    Methods
    -------
    get_geometry_matrix()
        Returns the geometry_matrix built by the class constructor.
    """

    def __init__(self, alpha, detector_cells, number_of_projections, object_size, omega_total=0):
        """
        It creates a new instance of the class InlineScanningSetup.
        :param alpha: fan-beam opening angle in the X-ray source;
        :param detector_cells: number of detector elements;
        :param number_of_projections: number of X-ray projections acquired during the object movement;
        :param object_size: number W of pixels of the W x W reconstruction grid;
        :param omega_total: total object rotation (in degrees) around its own axis between the first and the last projection
        acquisition.
        """
        self.det_size = detector_cells
        h = (detector_cells / 2) / tan(radians(alpha / 2))

        # src: the ray source
        srcX = np.linspace(-detector_cells / 2, detector_cells / 2, num=number_of_projections)
        srcY = np.linspace(h - object_size / 2, h - object_size / 2, num=number_of_projections)

        # d :  the center of the detector
        dX = np.linspace(-detector_cells / 2, detector_cells / 2, num=number_of_projections)
        #dY = np.linspace(-object_size / 2 , -object_size / 2, num=number_of_projections)
        dY = np.linspace(-100, -100, num=number_of_projections)

        # u :  the vector between the centers of detector pixels 0 and 1
        uX = np.linspace(1, 1, num=number_of_projections)
        uY = np.linspace(0, 0, num=number_of_projections)
        # pdb.set_trace()

        if omega_total > 0:

            omega = radians(omega_total / number_of_projections)

            for p in range(number_of_projections):
                retro = number_of_projections - p - 1

                alfa_zero = atan2(srcY[retro], srcX[retro])
                d = (srcY[retro] ** 2 + srcX[retro] ** 2) ** 0.5

                alfa_atual = p * omega + alfa_zero

                srcY[retro] = d * sin(alfa_atual)
                srcX[retro] = d * cos(alfa_atual)

                uY[p] = sin(p * omega)
                uX[p] = cos(p * omega)

        self.geometry_matrix = np.column_stack((srcX, srcY, dX, dY, uX, uY))

    def get_geometry_matrix(self):
        """
        It provides access to the inline CT setup built in the constructor method.
        :return: the attribute geometry_matrix
        """
        return self.geometry_matrix