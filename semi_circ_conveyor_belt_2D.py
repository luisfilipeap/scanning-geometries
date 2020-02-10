import numpy as np
from matplotlib import pyplot as plt
from math import radians, tan

class SemiCircularConveyorBelt:



    def __init__(self, radius, n_projs, src_dist, det_dist, fan_beam_angle):

        alpha = np.linspace(radians(60),radians(120), n_projs)

        srcX = (src_dist+radius)-(np.sin(alpha)*radius)
        srcY = - np.cos(alpha)*radius

        dx = (radius-det_dist) - np.sin(alpha)*radius
        dy = - np.cos(alpha)*radius

        ux = np.cos(-alpha)
        uy = np.sin(-alpha)

        self.geometry_matrix = np.column_stack((srcX, srcY, dx, dy, ux, uy))

        rad = radians(fan_beam_angle/2)
        self.det_size =  int(2*(tan(rad)*(radius+src_dist+det_dist)))


    def get_geometry_matrix(self):

        return self.geometry_matrix


    def get_det_size(self):

        return self.det_size



if __name__ == "__main__":
    g = SemiCircularConveyorBelt(15, 45, 5, 5, 30)

    alphas = np.linspace(0.1,1,45)
    rgba_colors_red = np.zeros((45,4))
    rgba_colors_red[:,0] = 1.0
    rgba_colors_red[:, 3] = alphas

    rgba_colors_blue = np.zeros((45,4))
    rgba_colors_blue[:,2] = 1.0
    rgba_colors_blue[:, 3] = alphas

    rgba_colors_green = np.zeros((45,4))
    rgba_colors_green[:,1] = 1.0
    rgba_colors_green[:, 3] = alphas

    plt.figure()
    plt.scatter(g.srcX, g.srcY, color=rgba_colors_red)
    plt.scatter(g.dx, g.dy, color=rgba_colors_green)
    plt.scatter(g.ux, g.uy, color=rgba_colors_blue)
    plt.show()







