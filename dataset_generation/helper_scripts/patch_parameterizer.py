import random

import numpy
import pymeshlab
import os
import gdist
import numpy as np
import igl
import multiprocessing as mp
import subprocess
import scipy
from math import ceil
import time
#from two_dimensional_procrustes import two_dimensional_procrustes_numpy
from scipy.linalg import orthogonal_procrustes

from scipy.stats import special_ortho_group


def slim(V, F, tid,it=200,uv_init = None):
    # fin = str(tid) + "i.obj"
    # fout = str(tid) + "o.obj"
    # igl.write_obj(fin,V,F)
    # os.system("..\ReweightedARAP.exe " + fin + " " + fout)
    # #subprocess.check_output(["ReweightedARAP.exe","tempi.obj tempo.obj"])
    # V,UV,_,F,_,_ = igl.read_obj(fout)
    # return UV

    assert (not np.isnan(V).any())
    assert (not np.isnan(F).any())
    ## Find the open boundary
    if uv_init is None:
        bnd = igl.boundary_loop(F)

        ## Map the boundary to a circle, preserving edge proportions
        bnd_uv = igl.map_vertices_to_circle(V, bnd)

        ## Harmonic parametrization for the internal vertices

        assert (not np.isnan(bnd).any())
        assert (not np.isnan(bnd_uv).any())
        uv_init = igl.harmonic_weights(V, F, bnd, bnd_uv, 1)

    if(np.isnan(uv_init).any()):
        print('got nan uv init, continuing')
        return None
    org_area = igl.doublearea(V,F)
    area = igl.doublearea(uv_init, F)
    if area[0]<0:
        area = - area
    if np.any(area/org_area<1e-6):
        print("init has flips, continuing")
        return None
    #slim = igl.SLIM(V, F, uv_init, bnd, bnd_uv, igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, 1.0e35)

    slim = igl.SLIM(V, F, uv_init, np.ones((1,1)),np.expand_dims(uv_init[0,:],0), igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, 1.0e1)
    slim.solve(it)
    uva = slim.vertices()
    uva -= uva.mean(axis = 0)
    # arap = igl.ARAP(V, F, 2, np.zeros(0))
    # uva = arap.solve(np.zeros((0, 0)), uv_init)

    return uva
# def slim(V, F, uv_init, bnd, bnd_uv, it):
#     slim = igl.SLIM(V, F, uv_init, bnd, bnd_uv, igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, 1.0e35)
#     slim.solve(it)
#     uva = slim.vertices()
#     return uva

## compute normals and remove unused vertices
def filter_mesh(V, F):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    mesh.add_vertex_scalar_attribute(np.arange(V.shape[0]), 'idx')
    ms.add_mesh(mesh)
    ms.remove_unreferenced_vertices()

    mesh = ms.current_mesh()

    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()
    V_idx = mesh.vertex_scalar_attribute_array('idx').astype(np.int64)

    return V_small, F_small, N_small, V_idx



## compute genus and area size
def compute_measures(V, F):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    ms.add_mesh(mesh)

    out_dict = ms.compute_geometric_measures()

    C = np.sqrt( np.pi / out_dict['surface_area'] )

    out_dict = ms.compute_topological_measures()

    return C, out_dict['number_holes']-1


# select vertex to keep (remove ears)
def filter_selection(input_mask,faces):
    # remove points\faces that are not fully selected, onlu faces that have 3 vertices selected
    face_mask = input_mask[faces]
    face_mask = face_mask.sum(axis=-1)
    keep_faces = (face_mask == 3)

    mask = np.zeros(input_mask.shape[0])
    for face_idx in np.nonzero(keep_faces)[0]:
        mask[faces[face_idx]] += 1.0
    mask = mask >= 2.0
    ### flap ears -> remove vertices that belong to only one face
    ### repeat (recursive)
    return mask, (mask == input_mask).all()


def parameterize(points, faces,threshold,max_vertices,sample_index,fname,tid,shared_list):
    min_vertices = 1000
    cur_threshold = threshold
    GROW_REGION = True
    if GROW_REGION:
        # CC = igl.face_components(faces)
        # if np.amax(CC)>0:
        #     print("more than one CC, continuing")
        #     return None
        if not igl.is_edge_manifold(faces):
            print("not edge manifold, returning")
            return None

    for param_attempts in range(10):
        #print("param" + str(param_attempts))
        for j in range(20):
            if GROW_REGION:
                uv_init = None
                #sv = np.nonzero(included_points == 0.0)[0]
                source_indices = np.random.choice(points.shape[0], 1).astype(np.int64)


                R = special_ortho_group.rvs(3)
                query_points = numpy.matmul(points,R)
                for i in range(3):
                    query_points[:,i] = query_points[:,i] * (0.6*i+1) #(1 + random.random()*2)
                # query_points = numpy.matmul(points,R.transpose())
                #print("getting distances")
                try:
                    #assert(numpy.amax(faces)<len(query_points))
                    #igl.write_obj("temp_tes.obj",query_points,faces)
                    distances = gdist.compute_gdist(query_points, faces.astype(np.int32), source_indices.astype(np.int32),
                                                target_indices=None, max_distance=threshold)
                except Exception as e:
                    print("EXCEPTION!!!")
                    print(e)


                #print("finished distances")
                for try_radius in range(100):
                    # compute geodesic distance to select points patch
                    mask = (distances >= -1.0e-6) * (distances < cur_threshold + 1.0e-6)
                    vnum = np.count_nonzero(mask)
                    if vnum < max_vertices:
                        break
                    # pi*r^2 area/tri_area, want pi*new_r^2/tri_area = 10000 --->
                    # pi*r^2/tri_area = K, then pi*r^2/pi*new_r^2 = K/10000 --->
                    # new_r = sqrt(10000/K)*r
                    red = np.sqrt(max_vertices / vnum) * 0.99
                    red = np.min([red, 0.9])
                    cur_threshold = cur_threshold * red
                else:
                    continue
                if vnum < min_vertices:
                    continue
                # remove ears from patch (vertex selection)

                mask, stop = filter_selection(mask,faces)
                while not stop:
                    mask, stop = filter_selection(mask,faces)
                #print("finish filtering")
                # select remaining faces from entire mesh faces
                face_mask = mask[faces]
                face_mask = face_mask.sum(axis=-1)
                keep_faces = face_mask > 2.0

                if not face_mask.any():
                    continue
                # remove unused vertices
                V_local, F_local, N_local, V_idx = filter_mesh(points, faces[keep_faces,:])

                # cuts = igl.cut_to_disk(F_local.astype(int))
                # if len(cuts)>0:
                #     bcuts = np.zeros(F_local.shape)
                #     for cut in cuts:
                #         for i in range(len(cut) - 1):
                #             found = np.logical_or(F_local == cut[i], F_local == cut[i+1])
                #             s = np.sum(found,1)
                #             assert(len(s) == bcuts.shape[0])
                #             ind = np.argwhere(s == 2)
                #             found[ind,:] = 0
                #             bcuts = np.logical_or(bcuts,found)
                #     V_local,F_local = igl.cut_mesh(V_local,F_local.astype(int),bcuts.astype(int))



                #C, genus = compute_measures(V_local, F_local)
                #cuts = igl.cut_to_disk(F_local)

                # if genus != 0:# or len(cuts) > 0:  # skip, patch need to be genus 0
                #     # selected_points = selected_points[:-1]
                #     # source_indices = np.random.choice(np.nonzero(included_points == 0.0)[0], 1).astype(np.int64)[0]
                #     # np.random.choice(points.shape[0], 1).astype(np.int64)
                #     source_indices = np.random.choice(points.shape[0], 1).astype(np.int64)  # np.array([source_indices])
                #     print("genus not zero, trying again")
                #     continue
                ec = igl.euler_characteristic(F_local)
                if ec != 1:
                    #print("genus not zero, trying again")
                    continue
            else:
                normals = igl.per_face_normals(points, faces, np.zeros(faces.shape))
                for k in range(50):
                    rand_normal = np.random.random((3,1)) - 0.5
                    norm = np.linalg.norm(rand_normal)
                    if norm == 0:
                        continue
                    rand_normal = rand_normal / norm
                    val = np.dot(normals, rand_normal).squeeze()
                    keep_faces = val > 0.01
                    if numpy.var(val[keep_faces])<0.001:
                        #print("flat region")
                        continue
                    MIN_TRI_NUM = 1000
                    MAX_TRI_NUM = 20000
                    if np.sum(keep_faces)<MIN_TRI_NUM or np.sum(keep_faces)>MAX_TRI_NUM:
                        #print("wrong face #1 " + str(np.sum(keep_faces)))
                        continue
                    if not keep_faces.any():
                        #print("wrong face")
                        continue
                    # remove unused vertices
                    V_local, F_local,_,_ = igl.remove_unreferenced(points, faces[keep_faces, :])

                    #CC_number, CC_indices,CC_size = igl.connected_components(igl.adjacency_matrix(F_local))
                    CC = igl.face_components(F_local)

                    values, counts = np.unique(CC, return_counts=True)
                    if np.max(counts) < MIN_TRI_NUM or np.sum(keep_faces)>MAX_TRI_NUM:
                        #print("wrong face #2")
                        continue
                    ind = np.argmax(counts)
                    max_ind = values[ind]
                    V_local, F_local,_,_= igl.remove_unreferenced(V_local, F_local[CC == max_ind, :])

                    frame = scipy.linalg.null_space(rand_normal.T)
                    uv_init = np.matmul(V_local, frame)
                    break
                else:
                    #print("giving up")
                    return


            print("parameterizing " + str(F_local.shape[0]) + " faces")

            # C is the normalization constant to resize the patch = to disk area
            C = 1/np.sum(igl.doublearea(V_local,F_local))
            V_local = C*np.array(V_local.tolist())
            #V_local = pca_embed(V_local)
            UV_local = slim(V_local, F_local, tid,uv_init = uv_init)
            if UV_local is None:
                continue
            if numpy.any(numpy.isnan(UV_local)):
                continue
            R,_ = orthogonal_procrustes(V_local,np.pad(UV_local,((0,0),(0,1))))

            V_local = V_local@R
            UV_local = np.hstack((UV_local, np.zeros((UV_local.shape[0], 1))))
            # bookkeeping (vertex that has been selected = 1)
            #included_points[mask] = 1.0

            # save patch as obj

            # write_mesh(os.path.join(output_folder, f'{file}_{i}.obj'), V_local, F_local, UV_local, None)
            sample_folder = fname+'_'+str(sample_index)+'_'
            shared_list.append((V_local,UV_local,F_local))
            return True
def thread_func(fnames,input_folder,K,sample_index,output_folder,thread_ind,shared_list):
    #print("Thread running on " +str(fnames))
    # read mesh
    threshold = 0.5 # geodesic distance to select the patch
    max_vertices = 6000
    count = 0
    NUM_SAMPLES = len(fnames)*K
    while(True):
        for file in fnames:
            fname = os.path.join(input_folder, file)
            if fname.endswith('.obj'):
                points, _, _, faces, _, _ = igl.read_obj(fname)
            elif fname.endswith('.off'):
                points, faces, _ = igl.read_off(fname)
            else:
                raise Exception("unknown extension")
            if points.shape[0] == 0:
                continue
            # for i in range(2):
            #     print(faces.shape)
            #     points, faces = igl.upsample(points, faces)
            dbl_area = igl.doublearea(points, faces)
            dbl_area = np.sum(dbl_area)
            points = points / np.sqrt(dbl_area)
            # pymeshlab.simplification_quadric_edge_collapse_decimation(points,faces,targetfacenum = 50000)
            #print('DONE loading mesh')
            C, genus = compute_measures(points, faces)
            points = points * C
            # memory of which vertices have been selected
            included_points = np.zeros(points.shape[0])
            # selected_points = []
            source_indices = np.array([0])
            N = points.shape[0]
            out_name = file.split('.')[0]
            out_name = os.path.join(output_folder,out_name)

            ret = parameterize(points, faces, threshold, max_vertices,sample_index+count,out_name,thread_ind,shared_list)
            print(f"*** thread {thread_ind}, {'suceeded' if ret else 'failed'} on {fname}/")

####################################################
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    manager = mp.Manager()
    shared_list = manager.list()
    input_folder = 'data/10k_surface'
    output_folder = input_folder + '_patches/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    files = [ el for el in os.listdir(input_folder) if el.endswith('obj') or el.endswith('off')]
    TOT_SAMPLES = 100000
    random.shuffle(files)
    K = ceil(TOT_SAMPLES/len(files))
    # files = files[:TOT_SAMPLES]


    N_THREADS = mp.cpu_count() -2
    sample_index = 0
    #files_lists = np.array_split(files,N_THREADS)
    threads = []
    for tind in range(N_THREADS): #enumerate(files_lists):
        files = files.copy()
        random.shuffle(files)
        threads.append(mp.Process(target=thread_func, args=(files,input_folder,1,sample_index,output_folder,tind,shared_list)))
        sample_index +=  len(files)
    for th in threads:
        th.start()
    count = 0
    pairs = []
    while(len(shared_list) == 0):
        pass
    while True:
        if len(shared_list)>0:
            V_local,UV_local,F_local = shared_list.pop(0)
            f1 = f'{count:08}'+'.obj'
            igl.write_obj(output_folder + f1, V_local, F_local)
            f2 = f'{count+1:08}'+'.obj'
            igl.write_obj(output_folder + f2, UV_local, F_local)
            pairs.append((f1,f2))
            print("********* writing pair # " +  str(int(count/2+1)) + '/' + str(TOT_SAMPLES) + '***********')
            count+=2
            if count/2 > TOT_SAMPLES:
                print("Reached dataset quota!!!!!!!!!!!!!!!!!!!!!!!!! exiting")
                break
        else:
            time.sleep(1)
            for i in range(len(threads)):
                if not threads[i].is_alive():
                    print(f"+++++++++++++++++++++++++++ thread {i} died, starting a new one! +++++++++++++++++++++++")
                    files = files.copy()
                    random.shuffle(files)
                    threads[i] = mp.Process(target=thread_func, args=(files, input_folder, 1, sample_index, output_folder, tind, shared_list))
                    threads[i].start()


    for th in threads:
        th.kill()
    # th.join()
    # for th in threads:
    #     th.join()
        #thread_func(files,input_folder,K,sample_index)
    import SourceToTargetsFile
    SourceToTargetsFile.write(output_folder + 'data.json', pairs)
    print('Exiting')


