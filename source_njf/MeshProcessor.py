from multiprocessing import process
import warnings
warnings.filterwarnings("ignore")


from scipy.sparse import load_npz, save_npz
from PoissonSystem import poisson_system_matrices_from_mesh, PoissonSystemMatrices, SparseMat
import os
import trimesh
from easydict import EasyDict
import numpy
import numpy as np
import scipy
import scipy.sparse
import igl
from scipy.sparse import save_npz
from time import time
import torch

NUM_SAMPLES = 1024
WKS_DIM = 100

class MeshProcessor:
    '''
    Extracts all preprocessing-related data (sample points  for pointnet; wave-kernel-signature, etc.)
    '''
    def __init__(self, vertices, faces, ttype, source_dir=None,from_file = False,
                 cpuonly=False, load_wks_samples=False, load_wks_centroids=False,
                compute_splu=True, load_splu=False):
        '''
        :param vertices:
        :param faces:
        :param ttype: the torch data type to use (float, half, double)
        :param source_dir: the directory to load the preprocessed data from; if given, will try to load the data before computing, if not given, always compute
        '''
        
        self.ttype = ttype
        self.num_samples = NUM_SAMPLES
        self.vertices = vertices.squeeze()
        self.faces = faces.squeeze()
        self.normals =  igl.per_vertex_normals(self.vertices, self.faces)
        # self.__use_wks = use_wks
        self.samples = EasyDict()
        self.samples.xyz = None
        self.samples.normals = None
        self.samples.wks = None
        self.centroids = EasyDict()
        self.centroids.points_and_normals = None
        self.centroids.wks = None
        self.diff_ops = EasyDict()
        self.diff_ops.splu = EasyDict()
        self.diff_ops.splu.L = None
        self.diff_ops.splu.U = None
        self.diff_ops.splu.perm_c = None
        self.diff_ops.splu.perm_r = None
        self.diff_ops.frames = None
        self.diff_ops.rhs = None
        self.diff_ops.grad = None
        self.diff_ops.poisson_sys_mat = None
        self.faces_wks = None
        self.vert_wks = None
        self.diff_ops.poisson = None
        self.source_dir = source_dir
        self.from_file = from_file
        self.cpuonly = cpuonly
        self.load_wks_samples = load_wks_samples
        self.load_wks_centroids = load_wks_centroids
        self.compute_splu = compute_splu
        self.load_splu = load_splu

    @staticmethod
    def meshprocessor_from_directory(source_dir, ttype, cpuonly=False, load_wks_samples=False, load_wks_centroids=False):
        try:
            vertices = np.load(os.path.join(source_dir, "vertices.npy"))
            faces = np.load(os.path.join(source_dir, "faces.npy"))
        except:
            print(os.path.join(source_dir, "vertices.npy"))
            import traceback
            traceback.print_exc()
        return MeshProcessor(vertices,faces,ttype,source_dir, cpuonly=cpuonly, load_wks_samples=load_wks_samples, load_wks_centroids=load_wks_centroids, compute_splu=False)

    @staticmethod
    def meshprocessor_from_file(fname, ttype, cpuonly=False, load_wks_samples=False, load_wks_centroids=False):
        if fname[-4:] == '.obj':
            V, _, _, F, _, _ = igl.read_obj(fname)
        elif fname[-4:] == '.off':
            V,F,_ = igl.read_off(fname)
        elif fname[-4:] == '.ply':
            V,F = igl.read_triangle_mesh(fname)
        return MeshProcessor(V,F,ttype,os.path.dirname(fname),True, cpuonly=cpuonly, load_wks_samples=load_wks_samples, load_wks_centroids=load_wks_centroids, compute_splu=False)

    @staticmethod
    def meshprocessor_from_array(vertices, faces, source_dir, ttype, cpuonly=False, load_wks_samples=False, load_wks_centroids=False):
        return MeshProcessor(vertices,faces,ttype,source_dir, cpuonly=cpuonly, load_wks_samples=load_wks_samples, load_wks_centroids=load_wks_centroids, compute_splu=False)



    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def load_centroids(self):
        self.centroids.points_and_normals = np.load(os.path.join(self.source_dir, "centroids_and_normals.npy"))
        if self.load_wks_centroids:
            self.centroids.wks = np.load(os.path.join(self.source_dir, "centroids_wks.npy"))

    def get_samples(self):
        if self.samples.xyz is None:
            if not self.from_file:
                try:
                    self.load_samples()
                except Exception as e:
                    self.compute_samples()
                    self.save_samples()
        return self.samples

    def load_samples(self):
        if self.samples.xyz is None:
            self.samples.xyz = np.load(os.path.join(self.source_dir, 'samples.npy'))
        if self.samples.normals is None:
            self.samples.normals = np.load(os.path.join(self.source_dir, 'samples_normals.npy'))
        if self.load_wks_samples:
            if  self.samples.wks is None:
                self.samples.wks = np.load(os.path.join(self.source_dir, 'samples_wks.npy'))
            if self.centroids.wks is None:
                self.centroids.wks = np.load(os.path.join(self.source_dir, 'centroid_wks.npy'))

    def save_samples(self):
        os.makedirs(self.source_dir, exist_ok=True)
        np.save(os.path.join(self.source_dir, 'samples.npy'), self.samples.xyz)
        np.save(os.path.join(self.source_dir, 'samples_normals.npy'), self.samples.normals)
        if self.load_wks_samples:
            np.save(os.path.join(self.source_dir, 'samples_wks.npy'), self.samples.wks)
            np.save(os.path.join(self.source_dir, 'centroid_wks.npy'), self.centroids.wks)

    def compute_samples(self):
        sstime = time()
        if self.load_wks_centroids or self.load_wks_centroids:
            self.computeWKS()
        # print(f"WKS {time() - sstime}")
        pt_samples, normals_samples, wks_samples, bary = self.sample_points( self.num_samples)
        self.samples.xyz = pt_samples
        self.samples.normals = normals_samples
        self.samples.wks = wks_samples
        self.centroids.wks = self.faces_wks

    def get_centroids(self):
        if self.centroids.points_and_normals is None:
            if not self.from_file:
                try:
                    self.load_centroids()
                except Exception as e:
                    self.compute_centroids()
                    # self.save_centroids() # centroid WKS and samples WKS are intertwined right now and you cannot really use one without the other. So this is redondont with function save_samples
        return self.centroids

    def compute_centroids(self):
        m = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.centroids.points_and_normals = np.hstack((np.mean(m.triangles, axis=1), m.face_normals))
        self.get_samples()# this is to compute WKS for centroids

    def get_differential_operators(self):
        if self.diff_ops.grad is None:
            if not self.from_file:
                try:
                    self.load_differential_operators()
                except Exception as e:
                    warnings.warn(f'while loading data, got file not exists exception: {e} ')
                    self.compute_differential_operators()
                    self.save_differential_operators()
        if self.load_splu:
            self.get_poisson_system()
        return self.diff_ops


    def load_poisson_system(self):
        try:
            self.diff_ops.splu.L = load_npz(os.path.join(self.source_dir, 'lap_L.npz'))
            self.diff_ops.splu.U = load_npz(os.path.join(self.source_dir, 'lap_U.npz'))
            self.diff_ops.splu.perm_c = np.load(os.path.join(self.source_dir, 'lap_perm_c.npy'))
            self.diff_ops.splu.perm_r = np.load(os.path.join(self.source_dir, 'lap_perm_r.npy'))
        except:
            print(f"FAILED load poisson on: {os.path.join(self.source_dir)}")
            raise Exception("FAILED load poisson on: {os.path.join(self.source_dir)}")


    def load_differential_operators(self): 
        self.diff_ops.rhs = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'new_rhs.npz')), ttype=torch.float64)
        self.diff_ops.grad = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'new_grad.npz')), ttype=torch.float64)
        self.diff_ops.frames = np.load(os.path.join(self.source_dir, 'w.npy'))
        self.diff_ops.laplacian = SparseMat.from_coo(load_npz(os.path.join(self.source_dir, 'laplacian.npz')), ttype=torch.float64)

    def save_differential_operators(self):  
        save_npz(os.path.join(self.source_dir, 'new_rhs.npz'), self.diff_ops.rhs.to_coo())
        save_npz(os.path.join(self.source_dir, 'new_grad.npz'), self.diff_ops.grad.to_coo())
        np.save(os.path.join(self.source_dir, 'w.npy'), self.diff_ops.frames)
        save_npz(os.path.join(self.source_dir, 'laplacian.npz'), self.diff_ops.laplacian.to_coo())

    def compute_differential_operators(self):
        '''
        process the given mesh
        '''
        poisson_sys_mat = poisson_system_matrices_from_mesh(V= self.vertices, F=self.faces,  cpuonly=self.cpuonly)
        self.diff_ops.grad = poisson_sys_mat.igl_grad
        self.diff_ops.rhs = poisson_sys_mat.rhs
        self.diff_ops.laplacian = poisson_sys_mat.lap
        self.diff_ops.frames = poisson_sys_mat.w
        self.diff_ops.poisson_sys_mat = poisson_sys_mat
    

    def compute_poisson(self):
        poissonsolver = poissonbuilder.compute_poisson_solver_from_laplacian(compute_splu=self.compute_splu)
        # new_grad = poissonbuilder.get_new_grad() # This is now done in poisson_system_matrices_from_mesh
        if self.compute_splu:
            self.diff_ops.splu.L, self.diff_ops.splu.U , self.diff_ops.splu.perm_c , self.diff_ops.splu.perm_r = poissonbuilder.compute_splu()
        self.diff_ops.frames = poissonbuilder.w


    def prepare_differential_operators_for_use(self,ttype):
        diff_ops = self.get_differential_operators() # call 1
        ## WARNING : we commented these two lines because they seemed redundant.
        if self.diff_ops.poisson_sys_mat is None: # not created if loaded from disk the diff ops
            diff_ops.poisson_sys_mat = PoissonSystemMatrices(self.vertices, self.faces, diff_ops.grad, diff_ops.rhs, diff_ops.frames, ttype, lap = diff_ops.laplacian,  cpuonly=self.cpuonly)
        
        self.diff_ops.poisson_solver = diff_ops.poisson_sys_mat.create_poisson_solver() # call 2

    def get_writeable(self):
        '''
        get dictionaries to write numpy and npz
        :return: two args, np, npz, each dicts with field_name --> data to save
        '''
        out_np = {}
        out_npz = {}
        out_np['vertices'] = self.vertices
        out_np['faces'] = self.faces
        if self.samples is not None:
            out_np["samples"] = self.samples.xyz
            out_np["samples_normals"] = self.samples.normals
            out_np["samples_wks"] = self.samples.wks
        if self.centroids is not None:
            out_np["centroids_wks"] = self.centroids.wks
            out_np["centroids_and_normals"] = self.centroids.points_and_normals
        if self.diff_ops is not None:
            out_np['lap_perm_c'] = self.diff_ops.splu.perm_c
            out_np['lap_perm_r'] = self.diff_ops.splu.perm_r
            out_np['w'] =self.diff_ops.frames
            out_npz['new_grad'] = self.diff_ops.grad.to_coo()
            out_npz['new_rhs'] = self.diff_ops.rhs
            out_npz['lap_L'] = self.diff_ops.splu.L
            out_npz['lap_U'] = self.diff_ops.splu.U
            out_npz['lap'] = self.diff_ops.poisson.lap
        return {key: value for key, value in out_np.items() if value is not None}, {key: value for key, value in out_npz.items() if value is not None}
        
    def get_data(self, key,file_type = 'npy'):
        if key == 'samples':
            return self.get_samples().xyz
        elif key == "samples_normals":
            return self.get_samples().normals
        elif key == "samples_wks":
            return self.get_samples().wks
        elif key == 'vertices':
            return self.vertices
        elif key == 'faces':
            return self.faces
        if file_type == 'npy':
            return np.load(os.path.join(self.source_dir, f'{key}.npy'))
        elif file_type == 'npz':
            return load_npz(os.path.join(self.source_dir, f'{key}.npz'))
        else:
            raise RuntimeError("wrong file type")

    def computeWKS(self):
        if self.faces_wks is  None or self.vert_wks is  None:
            st = time()
            w = WaveKernelSignature(self.vertices, self.faces, top_k_eig=50)
            w.compute()
            print(f"Ellapsed {time() - st}")
            wk = w.wks
            faces_wks = np.zeros((self.faces.shape[0], wk.shape[1]))
            for i in range(3):
                faces_wks += wk[self.faces[:, i], :]
            faces_wks /= 3
            self.faces_wks = faces_wks
            self.vert_wks = wk
            assert (self.faces_wks.shape[0] == self.faces.shape[0])
            assert (self.vert_wks.shape[0] == self.vertices.shape[0])


    def sample_points(self, n):
        bary, found_faces = igl.random_points_on_mesh(n, self.vertices, self.faces)
        vert_ind =  self.faces[found_faces]
        point_samples =  self.vertices[vert_ind[:,0]] * bary[:,0:1] + self.vertices[vert_ind[:,1]] * bary[:,1:2] + self.vertices[vert_ind[:,2]] * bary[:,2:3]
        normal_samples = self.normals[vert_ind[:,0]] * bary[:,0:1] + self.normals[vert_ind[:,1]] * bary[:,1:2] + self.normals[vert_ind[:,2]] * bary[:,2:3]
        wks_samples = None
        if self.load_wks_centroids or self.load_wks_samples:
            wks_samples = self.vert_wks[vert_ind[:,0]] * bary[:,0:1] + self.vert_wks[vert_ind[:,1]] * bary[:,1:2] + self.vert_wks[vert_ind[:,2]] * bary[:,2:3]
        return point_samples, normal_samples, wks_samples, bary

# This is insane to me
def sample_points(V, F, n):
    '''
    samples n points on the given mesh, along with normals and wks. Also return WKS of original faces (by averaging wks of 3 vertices of each face)
    :return:
    '''
    newF = F
    newV = V
    for iter in range(n):
        newV, newF = _sample_point(newV, newF)

    w = WaveKernelSignature(newV, newF, top_k_eig=100)
    w.compute()
    wk = w.wks
    sample_ks = wk[len(V):, :]
    org_ks = wk[:len(V), :]
    normals = igl.per_vertex_normals(newV, newF)
    normals = normals[len(V):, :]

    # get per-face wks by averaging its vertices
    faces_wks = np.zeros((F.shape[0], org_ks.shape[1]))
    for i in range(3):
        faces_wks += org_ks[F[:, i], :]
    faces_wks /= 3
    return newV[len(V):, :], normals, sample_ks, faces_wks


def _sample_point(VV, FF):
    while (True):
        bary, found_faces = igl.random_points_on_mesh(1, VV, FF)
        if (found_faces >= FF.shape[0]):
            continue
        # use to be 0.01
        if not numpy.any(bary < 0.05):
            break
    ret = numpy.zeros((1, VV.shape[1]))
    for i in range(VV.shape[1]):
        res = np.multiply(VV[FF[found_faces, :], i], bary)
        ret[:, i] = np.sum(res)
    newF = FF
    new_index = len(VV)
    new_tris = _insert_triangle(FF[found_faces, :], new_index)
    newF = numpy.concatenate((newF, new_tris), axis=0)
    newF = numpy.delete(newF, found_faces, axis=0)
    newV = numpy.concatenate((VV, ret), 0)
    return newV, newF


def _insert_triangle(old_tri, new_index):
    d = new_index
    a, b, c = (old_tri[0], old_tri[1], old_tri[2])
    new_tris = numpy.array([[a, b, d], [b, c, d], [c, a, d]])
    return new_tris





class WaveKernelSignatureError(Exception):
    pass

class WaveKernelSignature:
    '''
    Computes wave kernel signature for a given mesh
    '''
    def __init__(self,
                 vertices,
                 faces,
                 top_k_eig=200,
                 timestamps=WKS_DIM):
        # vertices, faces are both numpy arrays.
        self.vertices = vertices
        self.faces = faces

        # self.vertices_gpu = torch.from_numpy(vertices).cuda()
        # self.faces_gpu = torch.from_numpy(faces).cuda()

        self.top_k_eig = top_k_eig
        self.timestamps = timestamps
        self.max_iter = 10000

    def compute(self):
        '''
        compute the wks. Afterwards WKS stores in self.wks
        '''
        cp = igl.connected_components(igl.adjacency_matrix(self.faces))
        assert(cp[0]==1), f"{cp}"
        L = -igl.cotmatrix(self.vertices, self.faces) # this is fast 0.04 seconds
        M = igl.massmatrix(self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI)
        # assert(not numpy.any(numpy.isinf(L)))
        try:
            try:
                self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
                    L, self.top_k_eig, M, sigma=0, which='LM', maxiter=self.max_iter)
            except:
                self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
                    L, self.top_k_eig, M, sigma=1e-4, which='LM', maxiter=self.max_iter)                
        except:
            raise WaveKernelSignatureError("Error in computing WKS")

        # print(np.linalg.norm(self.eig_vecs, axis=0, keepdims=True))
        # print(np.max(self.eig_vecs))
        self.eig_vecs /= 200 #np.linalg.norm(self.eig_vecs, axis=0, keepdims=True)
        # np.save("norm_v2.npy", np.max(np.abs(self.eig_vecs), axis=0, keepdims=True))
        # np.save("norm_v2.npy", np.max(np.abs(self.eig_vecs), axis=0, keepdims=True))
        # print(np.linalg.norm(self.eig_vecs, axis=0))
        # print(np.max(self.eig_vecs, axis=0))
        # print(np.min(self.eig_vecs, axis=0))
        # print(self.eig_vals)

        # self.eig_vecs /= np.load('norm_v1.npy')

        # self.eig_vecs = self.eig_vecs / np.max(np.abs(self.eig_vecs), axis=0, keepdims=True)
        # self.eig_vecs = self.eig_vecs * np.load('norm_v2.npy')


        # nn = np.load('norm2.npy')
        # self.eig_vecs /= nn[:,:50]


        # ==== VISUALIZATION CODE ==========
        if False:
            num_mesh_to_viz = 6
            meshes = []
            for i in range(num_mesh_to_viz):
                meshes.append(trimesh.Trimesh(self.vertices + np.array([i*1,0,0]), self.faces, process=False))

            # mesh = meshes[0].union( meshes[1])
            # mesh = mesh.union( meshes[2])
            # mesh = mesh.union( meshes[3])
            meshes = [trimesh.util.concatenate(meshes)]

            from vedo import trimesh2vedo, show, screenshot, Plotter

            vp = Plotter(axes=0, offscreen=True)

            vmeshes = trimesh2vedo(meshes)
            cmaps = ('jet', 'PuOr', 'viridis')
            scals =  self.eig_vecs[:,:num_mesh_to_viz].transpose((1,0)).reshape(-1)
            vmeshes[0].cmap(cmaps[0], scals).lighting('plastic')

            # add a 2D scalar bar to a mesh
            vmeshes[0].addScalarBar(title=f"scalarbar #{0}", c='k') 

            vp.show(vmeshes, axes=1)
            screenshot(f"test_{time()}.png")
            import sys
            sys.exit(0)
        #  ================




        # range between biggest and smallest eigenvalue : 
        # 6 0.09622419080119388
        # 6_bis 0.09651935545457718
        delta = (np.log(self.eig_vals[-1]) - np.log(self.eig_vals[1])) / self.timestamps
        sigma = 7 * delta
        e_min = np.log(self.eig_vals[1]) + 2 * delta
        e_max = np.log(self.eig_vals[-1]) - 2 * delta
        es = np.linspace(e_min, e_max, self.timestamps)  # T
        self.delta = delta

        
        coef = np.expand_dims(es, 0) - np.expand_dims(np.log(self.eig_vals[1:]), 1)  # (K-1)xT
        coef = np.exp(-np.square(coef) / (2 * sigma * sigma))  # (K-1)xT #element wise square
        sum_coef = coef.sum(0)  # T
        K = np.matmul(np.square(self.eig_vecs[:, 1:]), coef)  # VxT. Scaling of the eigen vectors by coef. Coef depends only on the eigen values. Triangulation agnostic.
        self.wks = K / np.expand_dims(sum_coef, 0)  # VxT Scaling of the eigen vectors by sum_coef. Coef depends only on the eigen values. Triangulation agnostic.
        # print(np.linalg.norm(self.wks, axis=0))
        # print(np.linalg.norm(self.wks, axis=1))
        
