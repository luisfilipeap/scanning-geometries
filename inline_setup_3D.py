from math import floor, radians, tan, atan2, sin, cos
import numpy as np
import random

class InlineScanningSetup3D:
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

    def __init__(self, alpha, detector_cells, number_of_projections, object_size, vert_shift=0, tg_dir="esq", rotation=0):
        """
        It creates a new instance of the class InlineScanningSetup.
        :param alpha: fan-beam opening angle in the X-ray source;
        :param detector_cells: number of detector elements;
        :param number_of_projections: number of X-ray projections acquired during the object movement;
        :param object_size: number W of pixels of the W x W reconstruction grid;
        acquisition.
        """

        #h = (125) / tan(radians(alpha / 2))
        h = (detector_cells / 2) / tan(radians(alpha / 2))
        offset = -350



        # src: the ray source
        if tg_dir == "left":
            srcX = np.linspace(-offset-detector_cells/2, offset+detector_cells/2, num=number_of_projections)
            #srcX = np.linspace(-125, 125, num=number_of_projections)
        else:
            srcX = np.linspace(offset+detector_cells/2, -offset-detector_cells/2, num=number_of_projections)
            #srcX = np.linspace(125, -125, num=number_of_projections)

        z = h - object_size[2]/2
        srcZ = np.linspace(z*np.cos(np.deg2rad(rotation)), z*np.cos(np.deg2rad(rotation)), num=number_of_projections)
        srcY = np.linspace(vert_shift+z*np.sin(np.deg2rad(rotation)),vert_shift+z*np.sin(np.deg2rad(rotation)), num=number_of_projections)

        # d :  the center of the detector
        if tg_dir == "left":
            dX = np.linspace(-offset-detector_cells/2, offset+detector_cells/2, num=number_of_projections)
            #dX = np.linspace(-125,125, num=number_of_projections)
        else:
            dX = np.linspace(offset+detector_cells/2, -offset-detector_cells/2, num=number_of_projections)
            #dX = np.linspace(125, -125, num=number_of_projections)

        z = -object_size[2] / 2
        dZ = np.linspace(z*np.cos(np.deg2rad(rotation)), z*np.cos(np.deg2rad(rotation)), num=number_of_projections)
        dY = np.linspace(vert_shift+z*np.sin(np.deg2rad(rotation)), vert_shift+z*np.sin(np.deg2rad(rotation)), num=number_of_projections)

        # u :  the vector between the centers of detector pixels 0 and 1
        uX = np.linspace(1, 1, num=number_of_projections)
        uY = np.linspace(0, 0, num=number_of_projections)
        uZ = np.linspace(0, 0, num=number_of_projections)
        # pdb.set_trace()

        vX = np.linspace(0, 0, num=number_of_projections)
        vY = np.linspace(np.cos(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation)), num=number_of_projections)
        vZ = np.linspace(np.sin(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation)), num=number_of_projections)


        self.geometry_matrix = np.column_stack((srcX, srcY, srcZ,  dX, dY, dZ, uX, uY, uZ, vX, vY, vZ))

    def get_geometry_matrix(self):
        """
        It provides access to the inline CT setup built in the constructor method.
        :return: the attribute geometry_matrix
        """
        return self.geometry_matrix