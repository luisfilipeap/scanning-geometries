from inline_setup_3D import InlineScanningSetup3D
import astra
import numpy as np

class MultipleInlineContinuousScanningObject3D:

    def __init__(self, views_param, rec_size_param, n_proj_param, cells):
        self.S = 1
        self.cells = cells
        self.stages = []

        rot = 0
        vert_shift = -50#antes -250
        views = views_param
        for z in range(views):
            if z%2==0:
                dir = "left"
            else:
                dir = "right"
            self.stages.append(InlineScanningSetup3D(alpha=60, detector_cells=cells, number_of_projections=n_proj_param*self.S, object_size=rec_size_param, vert_shift=vert_shift, tg_dir=dir, rotation=rot))
            vert_shift = vert_shift + 25
            #rot = (z/views)*180


        self.geom_matrix = self.stages[0].get_geometry_matrix()
        for z in range(1, views):
            self.geom_matrix = np.concatenate((self.geom_matrix, self.stages[z].get_geometry_matrix()), axis=0)


        self.proj_geom = astra.create_proj_geom('cone_vec', cells, cells, self.geom_matrix)
        self.vol_geom = astra.create_vol_geom(rec_size_param[0], rec_size_param[1], rec_size_param[2])
        self.desired_projs = views*n_proj_param

    def run(self, phantom_param, n_iterations_param=700):

        id_old, proj_data = astra.create_sino3d_gpu(phantom_param, self.proj_geom, self.vol_geom)
        new_proj = np.zeros((self.cells, self.desired_projs, self.cells))

        for j in range(self.desired_projs):

            temp = np.sum(proj_data[:, j * self.S: j * self.S + self.S, :], axis=1)
            new_proj[:, j, :] = temp

        past_geom_matrix = self.geom_matrix
        self.new_geom_matrix = past_geom_matrix[range(int(self.S / 2), past_geom_matrix.shape[0], int(self.S)), :]
        new_geom = astra.create_proj_geom('cone_vec', self.cells, self.cells, self.new_geom_matrix)

        proj_id = astra.data3d.create('-sino', new_geom, new_proj)
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, n_iterations_param)

        output = {'rec': astra.data3d.get(rec_id), 'sino': new_proj}

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
        astra.projector.delete(id_old)

        return output


if __name__ == '__main__':

    import os
    from imageio import imread, imwrite

    views = 10
    p = 40
    #a = 60

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
        plane[:, :, k] = np.transpose(i)
        k = k + 1

    np.save('lamino_phantom.npy', plane)

    setup = MultipleInlineContinuousScanningObject3D(views_param = views, n_proj_param=p, rec_size_param=(565, 547, 187), cells=600 ) #antes800
    out = setup.run(plane)

    for k in range(187):
        if not os.path.isdir(dest + "\\{}-rows-{}-projs-continuous\\".format(views,p)):
            os.mkdir(dest + "\\{}-rows-{}-projs-continuous\\".format(views,p))
        final = out['rec']
        #final = (out['rec']-np.min(out['rec']))/(np.max(out['rec'])-np.min(out['rec']))
        imwrite(dest+"\\{}-rows-{}-projs-continuous\\slice{:05d}.png".format(views,p,k), final[k,:,:])

    for k in range(547):
        if not os.path.isdir(dest + "\\{}-rows-{}-projs-rec-continuous-2\\".format(views,p)):
            os.mkdir(dest + "\\{}-rows-{}-projs-rec-continuous-2\\".format(views,p))
        final = out['rec']
        #final = (out['rec']-np.min(out['rec']))/(np.max(out['rec'])-np.min(out['rec']))
        imwrite(dest+"\\{}-rows-{}-projs-rec-continuous-2\\slice{:05d}.png".format(views,p,k), final[:,:,k])

    if not os.path.isdir(dest+"\\{}-rows-{}-projs-sino-continuous\\".format(views,p)):
        os.mkdir(dest+"\\{}-rows-{}-projs-sino-continuous\\".format(views,p))
    for k in range(views*p):
        imwrite(dest+"\\{}-rows-{}-projs-sino-continuous\\proj_{:5d}.png".format(views,p,k),out['sino'][:,k,:])
