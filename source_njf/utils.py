import os
import shutil
import numpy as np
import torch

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def fix_orientation(vertices, faces):
    from igl import bfs_orient
    new_faces, c = bfs_orient(faces)
    new_faces = new_faces.astype(int)

    # Edge case: only one face
    if len(new_faces.shape) == 1:
        new_faces = new_faces.reshape(1,3)

    volume = signed_volume(vertices, new_faces)
    if volume < 0:
        new_faces = np.fliplr(new_faces)
    return new_faces

def signed_volume(v, f):
    # Add up signed volume of tetrahedra for each face
    # If triangles, then one of these vertices is the origin
    if f.shape[1] == 3:
        f = np.hstack([f, np.ones(len(f)).reshape(len(f), 1) * len(v)]).astype(int)
        v = np.vstack([v, np.zeros(3).reshape(1, 3)])
    fverts = v[f]
    fvectors = fverts - fverts[:,3, None,:]
    # Triple scalar product
    volume = 1/6 * np.sum(np.sum(fvectors[:,0,:] * np.cross(fvectors[:,1,:], fvectors[:,2,:], axis=1), axis=1))
    return volume

# Pytorch: from UVs back out Jacobian matrices per face using local coordinates per triangle
# NOTE: uvtri input must be PER triangle
def get_jacobian(uvtri, local_tris):
    x = local_tris[:,:,0]
    y = local_tris[:,:,1]

    # NOTE: Below only valid when local tris maps the first vertex to (0,0)!!!
    d = (x[:, 1] * y[:, 2]).reshape(len(x), 1)

    # Construct J
    Jx = torch.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]])
    Jy = torch.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]])
    J = torch.matmul(torch.stack([Jx, Jy], dim=1), uvtri) * 1/d # F x 2 x 2

    return J

# Get Tutte embedding from IGL
def tutte_embedding(vertices, faces):
    import igl
    bnd = igl.boundary_loop(faces)

    if bnd is None:
        raise ValueError(f"tutte_embedding: mesh has no boundary! set fixclosed = True to try to cut to disk topology.")

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    ## Harmonic parametrization for the internal vertices
    assert not np.isnan(bnd).any(), f"NaN found in boundary loop!"
    assert not np.isnan(bnd_uv).any(), f"NaN found in tutte initialized UVs!"
    uv_init = igl.harmonic_weights(vertices, faces, bnd, np.array(bnd_uv, dtype=vertices.dtype), 1)

    return uv_init

# Convert each triangle into local coordinates: A -> (0,0), B -> (x2, 0), C -> (x3, y3)
# TODO: general projection onto local bases
def get_local_tris(vertices, faces, basis=None, device=torch.device("cpu")):
    """basis: F x 1 array of integers from 1-6 indicating the basis type to initialize with
        (1) ab
        (2) ba
        (3) ac
        (4) ca
        (5) bc
        (6) cb """
    fverts = vertices[faces].to(device)
    if basis is None:
        e1 = fverts[:, 1, :] - fverts[:, 0, :]
        e2 = fverts[:, 2, :] - fverts[:, 0, :]
    else:
        ogdict = {0: 0, 1: 1, 2: 0, 3: 2, 4: 1, 5: 2}
        xdict = {0: 1, 1: 0, 2: 2, 3: 0, 4: 2, 5: 1}
        basisog = np.array([ogdict[b] for b in basis])
        basisx = np.array([xdict[b] for b in basis])
        used_i = np.stack([basisog, basisx], axis=1)
        basise = np.array([int(list(set(range(3)).difference(set(used)))[0]) for used in used_i]) # Set differences
        e1 = fverts[torch.arange(len(basisx)), basisx, :] - fverts[torch.arange(len(basisx)), basisog, :]
        e2 = fverts[torch.arange(len(basisx)), basise, :] - fverts[torch.arange(len(basisx)), basisog, :]

    # Vector parameters
    s = torch.linalg.norm(e1, dim=1).double()
    t = torch.linalg.norm(e2, dim=1).double()
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1)).double()

    if basis is None:
        x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)])
        y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)])
    else:
        # Position parameters
        x = torch.zeros(len(angle), 3).double()
        x[torch.arange(len(angle)), basisog] = s
        x[torch.arange(len(angle)),basise] = t * torch.cos(angle)

        y = torch.zeros(len(angle), 3).double()
        y[torch.arange(len(angle)), basise] = t * torch.sin(angle)

    local_tris = torch.stack((x, y), dim=-1).reshape(len(angle), 3, 2)
    return local_tris

# Return updated stitched soupvs, soupfs based on original topology (vs, fs) and epsilon
# NOTE: fs and soupfs MUST be in correspondence!
# NOTE: We are taking the original topology, and "cutting" wherever the edge distances are far apart
# TODO: KEEP DEBUGGING -- TUTTE INIT SHOULD RETURN NO CUTS
def stitchtopology(vs, fs, edgedist, epsilon=1e-2, return_cut_edges=False, return_cut_length = False):
    from meshing.mesh import Mesh
    from collections import defaultdict
    import copy
    # Loop over es => if edgedist < epsilon, merge vertex indices (keep same root index so all updates stay the same)
    ogmesh = Mesh(vs, fs)
    ogvs = vs
    newvs = []
    newfs = np.copy(fs)

    if return_cut_edges:
        cut_es = [ei for ei, edge in sorted(ogmesh.topology.edges.items()) if not edge.onBoundary()]

    if return_cut_length:
        cutlen = np.sum([ogmesh.length(e) for e in ogmesh.topology.edges.values() if not e.onBoundary()])

    # TODO: Keep track of all split vertex groupings, then deal with assignment after
    ### Merging algorithm
    # We first initialize everything assuming complete splits
    # Then merge groups back together
    facetogroup = {}
    vgroupn = defaultdict(int)
    splitgroups = defaultdict(lambda: defaultdict(list))

    # Initialize all splits
    for fi in range(len(fs)):
        f = fs[fi]
        facetogroup[fi] = []
        for v in f:
            v_gkey = vgroupn[v]
            facetogroup[fi].append(v_gkey)
            splitgroups[v][v_gkey].append(fi)
            vgroupn[v] += 1

    # Edge face correspondences
    fconn, vconn = ogmesh.topology.export_edge_face_connectivity(fs)

    # Merge groups together based on stitching criteria
    # NOTE: this is same order as export_edge_face_connectivity()
    boundarycount = 0
    for ei, edge in sorted(ogmesh.topology.edges.items()):
        if edge.onBoundary():
            boundarycount += 1
            continue
        if edgedist[ei - boundarycount] < epsilon: # NOTE: edgedist and edge/face correspondence REMOVE BOUNDARIES
            # Merge face groups together along the shared edge (merge into f0)
            f0, f1 = fconn[ei - boundarycount]
            f0_v1key = facetogroup[f0][vconn[ei - boundarycount][0][0]]
            f0_v2key = facetogroup[f0][vconn[ei - boundarycount][0][1]]
            f1_v1key = facetogroup[f1][vconn[ei - boundarycount][1][0]]
            f1_v2key = facetogroup[f1][vconn[ei - boundarycount][1][1]]

            # NOTE: We have to move ALL vertex groups associated with f1 edge
            # facetogroup[f1][vconn[ei - boundarycount][1][0]] = f0_v1key
            # facetogroup[f1][vconn[ei - boundarycount][1][1]] = f0_v2key

            v1 = edge.halfedge.vertex.index
            v2 = edge.halfedge.tip_vertex().index

            v1list = copy.copy(splitgroups[v1][f1_v1key])
            for fi in v1list:
                local_vi = None
                for i in range(3):
                    vi = fs[fi][i]
                    if vi == v1:
                        local_vi = i
                        break
                if local_vi is None:
                    raise ValueError(f"Face {fi} in split group under vertex {v1} but not found in original face indexing!")
                facetogroup[fi][local_vi] = f0_v1key
                splitgroups[v1][f1_v1key].remove(fi)
                splitgroups[v1][f0_v1key].append(fi)

            v2list = copy.copy(splitgroups[v2][f1_v2key])
            for fi in v2list:
                local_vi = None
                for i in range(3):
                    vi = fs[fi][i]
                    if vi == v2:
                        local_vi = i
                        break
                if local_vi is None:
                    raise ValueError(f"Face {fi} in split group under vertex {v2} but not found in original face indexing!")
                facetogroup[fi][local_vi] = f0_v2key
                splitgroups[v2][f1_v2key].remove(fi)
                splitgroups[v2][f0_v2key].append(fi)

            if return_cut_edges:
                cut_es.remove(ei)

            if return_cut_length:
                cutlen -= ogmesh.length(edge)

    ## Actually update topology
    for vi in range(len(vs)):
        for v_gkey, flist in splitgroups[vi].items():
            # New vertex for each splitgroup
            if len(flist) > 0:
                newvs.append(vs[vi])
                for f in flist:
                    for local_vi in range(3):
                        if fs[f][local_vi] == vi:
                            newfs[f][local_vi] = len(newvs)-1

    newvs = np.stack(newvs)

    # Unit tests
    newvs_count = np.sum([1 for group in splitgroups.values() for val in group.values() if len(val) > 0])
    assert newvs_count == len(newvs), f"Split log count: {newvs_count}. # new vs: {len(newvs)}"

    # No orphaned vertices
    assert np.all(np.arange(len(newvs)) == np.sort(np.unique(newfs))), f"Isolated vertices found!"

    newmesh = Mesh(newvs, newfs)

    # Deal with the fact that some split verties get re-merged
    # newvs, newfs, _ = newmesh.export_soup(remove_isolated_vertices=True)
    # newmesh = Mesh(newvs, newfs)
    # newvs = torch.from_numpy(newvs).float().to(vs.device)
    # newfs = torch.from_numpy(newfs).long().to(fs.device)

    retbundle = [newvs, newfs]

    if return_cut_edges:
        retbundle.append(cut_es)

    if return_cut_length:
        retbundle.append(cutlen)

    return retbundle

## Given collection of triangle soup with ground truth topology, compute least-squares translation per triangle to best align vertices
def leastSquaresTranslation(vertices, faces, trisoup, iterate=False, debug=False, patience=5,
                            return_cuts = False, cut_epsilon=1e-1, iter_limit=50, weightfaces=False):
    """ vertices: V x 3 np array
        faces: F x 3 np array
        trisoup: F x 3 x 2 np array

        returns: F x 2 numpy array with optimized translations per triangle"""
    # TODO: Update the edge correspondence code to take just the faces array and call edge_soup_correspondence!
    from meshing.mesh import Mesh
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(faces)
    mesh = Mesh(vertices, faces)

    uvsoup = trisoup.reshape(-1, 2)
    e1 = []
    e2 = []
    fconn = []
    nonbcount = 0
    for eidx, edge in sorted(mesh.topology.edges.items()):
        vkey = tuple(sorted([edge.halfedge.vertex.index, edge.halfedge.tip_vertex().index]))
        assert vkey in edgecorrespondences, f"Edge {vkey} not found in edgecorrespondences!"
        v = edgecorrespondences[vkey]

        if not edge.onBoundary():
            nonbcount += 1

        if len(v) > 1:
            v = edgecorrespondences[vkey]
            e1.extend(list(v[0]))
            e2.extend(list(v[1]))
            fconn.append(facecorrespondences[vkey])

    assert len(e1) == len(e2) == nonbcount * 2, f"Edge idx arrays should be twice the number of non-boundary edges! {len(e1)}  == {len(e2)} == {nonbcount}"
    ef0 = uvsoup[e1] # E*2 x 2
    ef1 = uvsoup[e2] # E*2 x 2
    fconn = np.array(fconn)

    A = np.zeros((len(ef0), len(faces)))

    # Duplicate every row of fconn (bc each edge involves two vertex pairs)
    edge_finds = np.repeat(fconn, 2, axis=0)
    A[np.arange(len(A)),edge_finds[:,0]] = 1 # distance vectors go f0 -> f1
    A[np.arange(len(A)),edge_finds[:,1]] = -1
    B = ef1 - ef0

    newsoup = np.copy(trisoup)
    if iterate:
        weights = np.ones((len(A), 1))
        finaltrans = np.zeros((len(trisoup), 1, 2))

        # Iterate until convergence (if edge distances stop updating)
        tot_count = 0
        stationary_count = 0
        prev_weights = None
        while True:
            if stationary_count >= patience:
                print(f"L0 convergence after {tot_count} steps!")
                break

            if tot_count >= iter_limit:
                print(f"Did not converge before iteration limit {iter_limit}!")
                break

            if not np.all(np.isfinite(weights)):
                # Replace nan with 0 and infty with 1
                print(f"Non-finite values discovered. Found {np.sum(np.isnan(weights))} nan. Found {np.sum(np.isinf(weights))} infty.")
                weights[np.isinf(weights)] = 1
                weights[np.isnan(weights)] = 0

            try:
                opttrans, residuals, rank, s = np.linalg.lstsq(weights * A, weights * B, rcond=None)
            except Exception as e:
                print(e)
                print(f"Error encountered in least squares, exiting...")
                break
                # import pdb
                # pdb.set_trace()
                # print(e)
                # raise Exception() from e
            finaltrans += opttrans.reshape(-1, 1, 2)
            newsoup = trisoup + finaltrans

            # If debugging, then visualize
            # if debug:
            #     ps_soup = ps.register_surface_mesh(f"soup {tot_count}", newsoup.reshape(-1, 2), np.arange(len(faces) * 3).reshape(-1, 3), edge_width=1,
            #                                        enabled=True)
            #     ps.show()

            # Recompute edge distances and associated weights
            newuvsoup = newsoup.reshape(-1, 2)
            ef0 = newuvsoup[e1] # E*2 x 2
            ef1 = newuvsoup[e2] # E*2 x 2
            B = ef1 - ef0 # E * 2 x 2

            # Weights are 1/(correspondence distances)
            weights = 1/(np.linalg.norm(B, axis=1, keepdims=1) + 1e-10)

            if weightfaces:
                faceweights = np.linalg.norm(ef1[mesh.ftoe] - ef0[mesh.ftoe], axis=-1) # F x 3 x 2
                weights = np.mean(faceweights, axis=[1,2])

            weights /= np.max(weights) # Normalize

            if debug:
                # Print distribution of weights
                print(np.quantile(weights, np.linspace(0,1,5)))

            # Stop condition edge distances don't update for more than 5 steps
            if prev_weights is not None:
                weight_deltas = np.mean(np.abs(prev_weights - weights))
                if weight_deltas <= 1e-2:
                    stationary_count += 1
                else:
                    stationary_count = 0
            prev_weights = weights
            tot_count += 1
            # print(f"L0 least squares translation: done with step {tot_count}. Stationary count: {stationary_count}.")
    else:
        opttrans, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        finaltrans = opttrans.reshape(-1, 1, 2)

    ## Return cutE x 2 x 3 array where each row is the vertex positions of the cut
    if return_cuts:
        cutlen = 0
        cutedges = []
        finalsoup = (trisoup + finaltrans).reshape(-1, 2)
        e1 = np.array(e1).reshape(-1, 2)
        e2 = np.array(e2).reshape(-1, 2)
        ef0 = finalsoup[e1] # E x 2 x 2
        ef1 = finalsoup[e2] # E x 2 x 2
        edge_delta = np.sum(np.linalg.norm(ef1 - ef0, axis=-1), axis=1) # E x 1
        bdrycount = 0
        for eidx, edge in sorted(mesh.topology.edges.items()):
            if edge.onBoundary():
                bdrycount += 1
                continue
            if edge_delta[eidx - bdrycount] > cut_epsilon:
                cutlen += mesh.length(edge)
                cutedges.append(eidx)

        return finaltrans, cutedges, cutlen

    return finaltrans

def polyscope_edge_perm(mesh):
    # Need to map edge ordering based on polyscope's scheme
    vs, fs, _ = mesh.export_soup()
    polyscope_edges = []
    for f in fs:
        for i in range(len(f)):
            e_candidate = {f[i], f[(i+1)%3]}
            if e_candidate not in polyscope_edges:
                polyscope_edges.append(e_candidate)
    mesh_edges = [set([v.index for v in e.two_vertices()]) for eidx, e in sorted(mesh.topology.edges.items())]
    # Build permutation
    edge_p = []
    for edge in polyscope_edges:
        found = 0
        for i in range(len(mesh_edges)):
            meshe = mesh_edges[i]
            if edge == meshe:
                edge_p.append(i)
                found = 1
                break
        if found == 0:
            raise ValueError(f"No match found for polyscope edge {edge}")
    return np.array(edge_p)

# ====================== UV Stuff ========================
class ZeroNanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad[grad != grad] = 0
        return grad

def new_topology_from_cuts(mesh, edgelist):
    """ Given unsorted list of edges to cut, sort the cuts by connections and make cuts """

def make_cut(mesh, cutlist):
    from meshing.edit import EdgeCut

    for i in range(len(cutlist)-1):
        vsource = cutlist[i]
        vtarget = cutlist[i+1]

        ## Unit tests
        # Source vertex should be on boundary
        assert mesh.topology.vertices[vsource].onBoundary()

        for he in mesh.topology.vertices[vsource].adjacentHalfedges():
            if he.tip_vertex().index == vtarget:
                break

        try:
            edt = EdgeCut(mesh, he.edge.index, vsource, cutbdry=True)
            edt.apply()
            del edt
        except Exception as e:
            print(e)
            return False

    return True

def generate_random_cuts(mesh, enforce_disk_topo=True, diskmode='SHORTEST', max_cuts=10, verbose=False):
    """ Generate random cuts on mesh -- enforce disk topology if necessary

    Args:
        mesh (Mesh): halfedge mesh data structure
        enforce_disk_topo (bool, optional): Whether to enforce disk topology. Defaults to True.
        diskmode (str, optional): Type of path to cut to disk. Options {SHORTEST, RANDOM}. Defaults to 'SHORTEST'.
    """
    from meshing.edit import EdgeCut

    cutvs = []
    while (max_cuts - len(cutvs)) > 0:
        valid = False
        patience = 10
        while not valid:
            edgei = np.random.randint(0, len(mesh.topology.edges))
            edge = mesh.topology.edges[edgei]

            if patience <= 0:
                break

            if edge.onBoundary():
                patience -= 1
                continue

            # Having both vertices on boundary is okay as long as they are different boundaries!
            if edge.halfedge.vertex.onBoundary() and edge.halfedge.twin.vertex.onBoundary():
                v1bd = None
                for he in edge.halfedge.vertex.adjacentHalfedges():
                    if he.onBoundary:
                        v1bd = he.face
                        break
                v2bd = None
                for he in edge.halfedge.twin.vertex.adjacentHalfedges():
                    if he.onBoundary:
                        v2bd = he.face
                        break
                if v1bd is None or v2bd is None:
                    raise ValueError("Boundary topology is bugged! Vertices on boundary but no halfedge boundary found.")
                valid = v1bd != v2bd
            else:
                valid = True

            if not valid:
                patience -= 1

        if patience <= 0:
            print(f"No more valid edges found after 10 samples! {len(cutvs)} cuts generated.")
            break

        # Having both vertices on boundary is okay as long as they are different boundaries!
        if edge.halfedge.vertex.onBoundary() and edge.halfedge.twin.vertex.onBoundary():
            sourcev = edge.halfedge.vertex.index
            targetv = edge.halfedge.twin.vertex.index

            cut_vs = [mesh.vertices[sourcev], mesh.vertices[targetv]]
            cutvs.append(np.stack(cut_vs))

            # Visualize the cuts
            # ps.remove_all_structures()
            # ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices)
            # cutes = np.array([[i, i+1] for i in range(0, len(cutvs)-1)])
            # currentcut = np.array(cutvs)
            # ps_curve = ps.register_curve_network("cut", currentcut, cutes, enabled=True)
            # ps.show()

            EdgeCut(mesh, edgei, sourcev, cutbdry=True).apply()

        # If vertex is starting on boundary, then this is simple cut case
        elif edge.halfedge.vertex.onBoundary() or edge.halfedge.twin.vertex.onBoundary():
            sourcev = edge.halfedge.vertex.index if edge.halfedge.vertex.onBoundary() else edge.halfedge.twin.vertex.index
            targetv = edge.halfedge.vertex.index if edge.halfedge.twin.vertex.onBoundary() else edge.halfedge.twin.vertex.index

            cut_vs = [mesh.vertices[sourcev], mesh.vertices[targetv]]
            cutvs.append(np.stack(cut_vs))

            # Visualize the cuts
            # ps.remove_all_structures()
            # ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices)
            # cutes = np.array([[i, i+1] for i in range(0, len(cutvs)-1)])
            # currentcut = np.array(cutvs)
            # ps_curve = ps.register_curve_network("cut", currentcut, cutes, enabled=True)
            # ps.show()

            EdgeCut(mesh, edgei, sourcev).apply()

        else:
            # Need to sample a second edge (not on boundary)
            e2_candidates = [e.index for e in edge.halfedge.vertex.adjacentEdges() if not e.onBoundary() and e != edge] + \
                        [e.index for e in edge.halfedge.twin.vertex.adjacentEdges() if not e.onBoundary() and e != edge]
            if len(e2_candidates) == 0:
                continue
            else:
                e2_i = np.random.choice(e2_candidates)

            # Sourcev is shared vertex between the two edges
            presourcev = edge.halfedge.twin.vertex
            sourcev = edge.halfedge.vertex
            otheredge = mesh.topology.edges[e2_i]
            targetv = otheredge.halfedge.vertex
            if sourcev not in mesh.topology.edges[e2_i].two_vertices():
                sourcev = edge.halfedge.twin.vertex
                presourcev = edge.halfedge.vertex
            if targetv in edge.two_vertices():
                targetv = otheredge.halfedge.twin.vertex
            assert sourcev in mesh.topology.edges[e2_i].two_vertices()
            assert presourcev not in mesh.topology.edges[e2_i].two_vertices()
            assert targetv not in edge.two_vertices()
            sourcev = sourcev.index
            targetv = targetv.index
            presourcev = presourcev.index

            cut_vs = [mesh.vertices[presourcev], mesh.vertices[sourcev], mesh.vertices[targetv]]
            cutvs.append(np.stack(cut_vs))

            # Visualize the cuts
            # ps.remove_all_structures()
            # ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices)
            # cutes = np.array([[i, i+1] for i in range(0, len(cutvs)-1)])
            # currentcut = np.array(cutvs)
            # ps_curve = ps.register_curve_network("cut", currentcut, cutes, enabled=True)
            # ps.show()

            # Actual edge should be second edge
            edge = mesh.topology.edges[e2_i]

            EdgeCut(mesh, edgei, sourcev, cutbdry=True, e2_i=e2_i).apply()

    # Check for disk topology condition
    if enforce_disk_topo and not mesh.is_disk_topology():
        print("Cut mesh is not disk topology. Generating cuts to enforce disk topology.")
        cutvs.extend(cut_to_disk(mesh, mode=diskmode, limit=1000))

    return cutvs

def cut_to_disk(mesh, mode='SHORTEST', verbose=False, limit=10):
    """ Cuts mesh to topological disk if necessary

    Args:
        mesh (Mesh): halfedge mesh data structure
        mode (str, optional): Type of path to cut. Options {SHORTEST, RANDOM}. Defaults to 'SHORTEST'.
    """
    from meshing.edit import EdgeCut
    count = 0
    cutvs = []

    # Don't allow cut if mesh has isolated faces
    if mesh.topology.hasIsolatedFaces():
        print("Mesh has isolated faces! Cannot cut to disk.")
        return cutvs

    # If mesh has no boundaries, then cut two edges to make a boundary
    if len(mesh.topology.boundaries) == 0:
        ei = np.random.choice(len(mesh.topology.edges))
        edge = mesh.topology.edges[ei]
        sourcev = edge.halfedge.tip_vertex()
        sourcevi = sourcev.index
        ei2 = np.random.choice(list(e.index for e in sourcev.adjacentEdges() if e != edge))

        # Cut
        EdgeCut(mesh, ei, sourcevi, e2_i= ei2).apply()

        # Get second edge
        edge2 = mesh.topology.edges[ei2]
        v1, v2 = edge2.two_vertices()
        sourcev2 = v1 if v2 == sourcev else v2
        edge3 = None
        for e in sourcev2.adjacentEdges():
            v1, v2 = e.two_vertices()
            if e.index != ei2 and not e.onBoundary() and (not v1.onBoundary() or not v2.onBoundary()):
                edge3 = e
                break

        # Debugging: visualize cut
        # import polyscope as ps
        # ps.init()
        # ps.remove_all_structures()
        # vs, fs, es = mesh.export_soup()
        # ps_mesh = ps.register_surface_mesh("mesh", vs, fs)
        # ps_mesh.set_edge_permutation(polyscope_edge_perm(mesh))
        # ecolors = np.zeros(len(es))
        # ecolors[[ei, ei2, edge3.index]] = 1
        # ps_mesh.add_scalar_quantity("cut es", ecolors, defined_on='edges', enabled=True)
        # ps.show()

        EdgeCut(mesh, edge3.index, sourcev2.index).apply()

        tmpcuts = mesh.vertices[[edge.halfedge.vertex.index, sourcevi, sourcev2.index]]
        cutvs.append(tmpcuts)

    while len(mesh.topology.boundaries) > 1:
        if verbose:
            import time
            t0 = time.time()

        # Draw cut starting from longest boundary to nearest boundary -> repeat until only 1 boundary left
        # Get longest boundary
        current_b = 0
        max_b_length = 0
        for b in mesh.topology.boundaries.values():
            b_edge_vs = np.array([list(v.index for v in e.two_vertices()) for e in b.adjacentEdges()])
            b_v_pos = mesh.vertices[b_edge_vs]
            b_length = np.sum(np.linalg.norm(b_v_pos[:,0,:] - b_v_pos[:,1,:], axis=1))
            if b_length > max_b_length:
                current_b = b.index
                max_b_length = b_length

        # Get closest boundary to current boundary from current cut point
        current_boundary = mesh.topology.boundaries[current_b]
        avail_b = list(k for k in mesh.topology.boundaries.keys() if k != current_b)
        subboundary_vs = np.array([v.index for v in current_boundary.adjacentVertices()])

        import igraph as ig
        vs, fs, es = mesh.export_soup()
        edgeweights = [mesh.length(e) for e in mesh.topology.edges.values()]
        graph = ig.Graph(len(vs), es)

        b_vs = np.array([v.index for b in avail_b for v in mesh.topology.boundaries[b].adjacentVertices()])

        if len(b_vs) == 0:
            print(f"Overlapping boundaries!")
            break

        if verbose:
            print(f"Iteration {count}: graph construct time {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        if mode == "SHORTEST":
            # Heuristic: initialize to first vertex in subboundary, and compute all shortest paths to all other boundaries
            # Choose shortest path and cut
            cutlists = []
            for init_v in subboundary_vs:
                tmpcutlists = graph.get_shortest_paths(init_v, b_vs, edgeweights)
                if len(tmpcutlists) > 0:
                    cutlists.extend(tmpcutlists)
            # All cut lists must be at least length 2
            cutlists = [cutlist for cutlist in cutlists if len(cutlist) >= 2]
            if len(cutlists) == 0:
                print("No more paths found.")
                return cutvs

            cutlens = [len(cut) for cut in cutlists]
            cutlist = cutlists[np.argmin(cutlens)]
            if verbose:
                print(f"Iteration {count}: shortest path calc {time.time() - t0:0.2f} sec.")
                print(f"\tCutlist {cutlist}. # boundaries: {len(mesh.topology.boundaries)}")
                t0 = time.time()

            # Get boundary of target
            shortest_target = cutlist[-1]
            for b in avail_b:
                if shortest_target in [v.index for v in mesh.topology.boundaries[b].adjacentVertices()]:
                    next_b = b
                    break

        elif mode == "RANDOM":
            random_source = np.random.choice(subboundary_vs)
            random_target = np.random.choice(b_vs)

            allpaths = list(graph.get_all_simple_paths(random_source, random_target))

            # Restrict to valid paths (length >= 2)
            allpaths = [path for path in allpaths if len(path) >= 2]

            if len(allpaths) == 0:
                print("No valid random cut paths found.")
                return cutvs

            cutlist = allpaths[np.random.choice(len(allpaths))]

        success = make_cut(mesh, cutlist)
        count += 1
        cutvs.append(mesh.vertices[cutlist])

        if verbose:
            print(f"Iteration {count}: cutting took {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        if not success:
            print(f"Disk cutting not successful!")
            return cutvs

        if count >= limit:
            print(f"# Boundaries exceeded limit {limit}!")
            return cutvs

    return cutvs

def cut_to_disk_single(mesh, singular_vs, verbose=False):
    count = 0

    for target_v in singular_vs:
        if verbose:
            import time
            t0 = time.time()

        # Build weighted edgelist
        weighted_edgelist = [[v.index for v in e.two_vertices()] + [mesh.length(e)] for \
                                e in mesh.topology.edges.values()]

        import igraph as ig
        vs, fs, es = mesh.export_soup()
        edgeweights = [mesh.length(e) for e in mesh.topology.edges.values()]
        graph = ig.Graph(len(vs), es)

        # Compute shortest paths from current target vertex to all other boundary vertices
        b_vs = np.array([v.index for b in mesh.topology.boundaries.values() for v in b.adjacentVertices()])
        cutlists = graph.get_shortest_paths(target_v, b_vs, edgeweights)

        if len(cutlists) == 0:
            print("No path found from vertex to boundary.")
            continue

        cutlens = [len(cut) for cut in cutlists]
        cutlist = cutlists[np.argmin(cutlens)]
        if verbose:
            print(f"Iteration {count}: shortest path calc {time.time() - t0:0.2f} sec.")
            print(f"\tCutlist {cutlist}. # boundaries: {len(mesh.topology.boundaries)}")
            t0 = time.time()

        # Generate cut
        make_cut(mesh, cutlist)
        count += 1

        graph.clear()
        del graph

        if mesh.topology.hasNonManifoldEdges():
            print(f"Mesh became non-manifold from cuts!")
            break

def SLIM(mesh, iters=500, v_with_holes = None, f_with_holes = None):
    # SLIM parameterization
    # Initialize using Tutte embedding
    import igl
    from meshing.mesh import Mesh

    vs, fs, _ = mesh.export_soup()
    uv_init = tutte_embedding(vs, fs)

    # Need to subset back non-disk topology if filled hole
    if v_with_holes is not None and f_with_holes is not None:
        # Only select UVs relevant to the
        uv_init = uv_init[v_with_holes]
        sub_faces = fs[f_with_holes]
        # NOTE: vertices should now be indexed in the same way as the original sub_faces
        submesh = Mesh(uv_init, sub_faces)
        vs, fs, _ = submesh.export_soup()

    slim = igl.SLIM(vs, fs, uv_init, np.ones((1,1)),np.expand_dims(uv_init[0,:],0), igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, 1.0e1)
    slim.solve(iters)
    slim_uv = slim.vertices()
    slim_uv -= slim_uv.mean(axis = 0)
    return slim_uv, slim.energy()

# Duplicates a vertex for each disjoint fan it belongs to
# Use to solve problem of nonmanifold vertices
def cut_vertex(mesh, vind):
    # Find all halfedges that contain the vertex
    heset = set()
    for he in mesh.topology.halfedges.values():
        if he.vertex.index == vind:
            heset.update([he.index])

    # Remove original vertex adjacent halfedge set
    heset = heset.difference(set([he.index for he in mesh.topology.vertices[vind].adjacentHalfedges()]))
    while len(heset) > 0:
        startind = heset.pop()
        starthe = mesh.topology.halfedges[startind]
        currenthe = mesh.topology.halfedges[startind]

        # Duplicate and add new vertex to mesh
        mesh.vertices = np.append(mesh.vertices, [mesh.vertices[vind]], axis=0)
        newv = mesh.topology.vertices.allocate()
        newv.halfedge = currenthe
        assert newv.index == len(mesh.vertices) - 1
        visited = set([currenthe.index])
        while currenthe.twin.next != starthe:
            currenthe.vertex = newv
            currenthe = currenthe.twin.next
            visited.update([currenthe.index])
        currenthe.vertex = newv

        heset = heset.difference(visited)

def run_slim(mesh, cut=True, verbose=False, time=False, max_boundaries=20):
    did_cut = False
    if mesh.topology.hasNonManifoldEdges():
        print(f"run_slim: Non-manifold edges found.")
        return None, None, did_cut

    if cut:
        if time:
            import time
            t0 = time.time()

        # Check for nonmanifold vertices while only one boundary
        # if mesh.topology.hasNonManifoldVertices():
        #     print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
        #     for vind in mesh.topology.nonmanifvs:
        #         cut_vertex(mesh, vind)

        if len(mesh.topology.boundaries) == 0 or len(mesh.topology.boundaries) > 1:
            cut_to_disk(mesh, mode='SHORTEST', verbose=verbose, limit=max_boundaries)
            did_cut = True

        # Check for nonmanifold vertices while only one boundary
        # if mesh.topology.hasNonManifoldVertices():
        #     print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
        #     for vind in mesh.topology.nonmanifvs:
        #         cut_vertex(mesh, vind)

        # if not hasattr(mesh, "vertexangle"):
        #     from meshing.analysis import computeVertexAngle
        #     computeVertexAngle(mesh)

        # Cut cones
        # Only cut cones if one boundary exists
        # singlevs = np.where(2 * np.pi - mesh.vertexangle >= np.pi/2)[0]
        # if len(singlevs) >= 0 and len(mesh.topology.boundaries) == 1: # Edge case: no boundaries
        #     cut_to_disk_single(mesh, singlevs, verbose)
        #     did_cut = True

        # Check for nonmanifold vertices while only one boundary
        # if mesh.topology.hasNonManifoldVertices():
        #     print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
        #     for vind in mesh.topology.nonmanifvs:
        #         cut_vertex(mesh, vind)

        # Don't parameterize nonmanifold after cut
        if mesh.topology.hasNonManifoldEdges():
            print(f"run_slim: Cut mesh has nonmanifold edges.")
            return None, None, did_cut

        if time:
            import time
            print(f"Cutting took {time.time() - t0:0.3f} sec.")

    # Compute SLIM
    try:
        uvmap, energy = SLIM(mesh)
    except Exception as e:
        print(e)
        return None, None, did_cut

    assert len(uvmap) == len(mesh.vertices), f"UV: {uvmap.shape}, vs: {mesh.vertices.shape}"

    return uvmap, energy, did_cut

# Get Jacobian of UV map
def get_jacobian_torch(vs, fs, uvmap, device=torch.device('cpu')):
    # Visualize distortion
    from igl import grad
    import torch
    G = torch.from_numpy(np.array(grad(vs.detach().cpu().numpy(), fs.detach().cpu().numpy()).todense())).to(device)

    # NOTE: currently gradient is organized as X1, X2, X3, ... Y1, Y2, Y3, ... Z1, Z2, Z3 ... resort to X1, Y1, Z1, ...
    splitind = G.shape[0]//3
    newG = torch.zeros(G.shape, device=device, dtype=uvmap.dtype)
    newG[::3] = G[:splitind]
    newG[1::3] = G[splitind:2*splitind]
    newG[2::3] = G[2*splitind:]

    J = (newG @ uvmap).reshape(-1, 3, 2).transpose(2,1) # F x 2 x 3
    return J

# Get Jacobian of UV map
def get_jacobian(vs, fs, uvmap):
    # Visualize distortion
    from igl import grad
    G = grad(vs, fs).todense()

    # NOTE: currently gradient is organized as X1, X2, X3, ... Y1, Y2, Y3, ... Z1, Z2, Z3 ... resort to X1, Y1, Z1, ...
    splitind = G.shape[0]//3
    newG = np.zeros_like(G)
    newG[::3] = G[:splitind]
    newG[1::3] = G[splitind:2*splitind]
    newG[2::3] = G[2*splitind:]

    from scipy import sparse
    newG = sparse.csc_matrix(newG)
    J = (newG @ uvmap).reshape(-1, 3, 2).transpose(0,2,1) # F x 2 x 3
    return J

### Get edge correspondences of soup
# Assumes we are indexing some node array V by taking V[fs] and then flattening
def edge_soup_correspondences(fs):
    from collections import defaultdict
    edgecorrespondences = defaultdict(list) # {v1, v2} (original topology) => [(v1a, v2a), (v1b, v2b)] (soup vertices indexing F*3 x 3)
    facecorrespondences = defaultdict(list) # {v1, v2} (original topology) => [f1, f2] (soup faces corresponding with edge corrs)
    for fi in range(len(fs)):
        f = fs[fi]
        for i in range(3):
            v1, v2 = f[i], f[(i+1)%3]
            if v1 > v2:
                edgekey = (v2, v1)

                # Corresponding soup index
                soupkey = (fi * 3 + (i+1)%3, fi * 3 + i)

            else:
                edgekey = (v1, v2)

                # Corresponding soup index
                soupkey = (fi * 3 + i, fi * 3 + (i+1)%3)

            currentlist = edgecorrespondences[edgekey]
            if len(currentlist) == 2:
                raise ValueError("There should only be two edge correspondences per edge!")

            # Find the corresponding index in the tutte vertex soup
            currentlist.append(soupkey)

            currentlist = facecorrespondences[edgekey]
            if len(currentlist) == 2:
                raise ValueError("There should only be two face correspondences per edge!")

            # Find the corresponding index in the tutte vertex soup
            currentlist.append(fi)

    return edgecorrespondences, facecorrespondences

## Original vertices to soup vertices
def vertex_soup_correspondences(fs):
    from collections import defaultdict
    vcorrespondence = defaultdict(list) # {ogv} (original topology) => [list of soup vertices] (soup vertices indexing F*3 x 3)
    for fi in range(len(fs)):
        f = fs[fi]
        for i in range(3):
            currentlist = vcorrespondence[f[i]]

            # Find the corresponding index in the tutte vertex soup
            currentlist.append(fi * 3 + i)

    return vcorrespondence

## From a given topology with soup pairs, get only the pairs which correspond to valid edges (i.e. incident to edges which are shared between faces)
## Returns the valid pairs, the corresponding indices (into the original pair array), and the corresponding edges lengths
def get_edge_pairs(mesh, valid_pairs, device=torch.device('cpu')):
    """ valid_pairs_edges: valid pairs which correspond to edges
        valid_pairs_to_soup_edges: for each valid pair, gives corresponding soup edge (two pairs of vertex indices).
        edgeidxs: indexes into the valid pairs array to get the ones corresponding to edges
        elens: edge lengths of the valid pairs """
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(mesh.faces)
    checkpairs = [set(pair) for pair in valid_pairs]

    # Construct pairs from original topology based on ordering from valid_pairs
    facepairs = []
    edgeidxs = []
    edgededupidxs = []
    valid_pairs_edges = []
    valid_pairs_to_soup_edges = []
    for i in range(len(checkpairs)):
        pair = checkpairs[i]
        found = False
        ei = 0
        for k, v in edgecorrespondences.items():
            # Skip boundary edges
            if len(v) == 2:
                if pair == set([v[0][0], v[1][0]]) or pair == set([v[0][1], v[1][1]]):
                    facepairs.append(facecorrespondences[k])
                    valid_pairs_to_soup_edges.append(v) # NOTE: This will be TWO pairs (one for each edge/soupface)
                    valid_pairs_edges.append(list(pair))
                    edgeidxs.append(i)
                    edgededupidxs.append(ei)
                    found = True
                    break
            ei += 1

    # Get edge lengths corresponding to the face pairs
    # NOTE: Edge lengths are already normalized by the mesh normalization
    edges = []
    elens = []
    for fpair in facepairs:
        founde = None
        for e in mesh.topology.faces[fpair[0]].adjacentEdges():
            if e.halfedge.face.index == fpair[1] or e.halfedge.twin.face.index == fpair[1]:
                founde = e
                edges.append(e.index)
                break
        if not founde:
            raise ValueError(f"Face pair {fpair} not found in mesh topology!")
        elens.append(mesh.length(founde))
    elens = torch.tensor(elens, device=device)

    assert len(elens) == len(edgeidxs), f"Elens must be equal to adjacent ids found in valid pairs! {len(elens)} != {len(edgeidxs)}"

    return valid_pairs_edges, valid_pairs_to_soup_edges, edgeidxs, edgededupidxs, edges, elens, facepairs

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort).double()  # for sape

    def forward(self, x):
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)

def signed_volume(v, f):
    # Add up signed volume of tetrahedra for each face
    # If triangles, then one of these vertices is the origin
    if f.shape[1] == 3:
        f = np.hstack([f, np.ones(len(f)).reshape(len(f), 1) * len(v)]).astype(int)
        v = np.vstack([v, np.zeros(3).reshape(1, 3)])
    fverts = v[f]
    fvectors = fverts - fverts[:,3, None,:]
    # Triple scalar product
    volume = 1/6 * np.sum(np.sum(fvectors[:,0,:] * np.cross(fvectors[:,1,:], fvectors[:,2,:], axis=1), axis=1))
    return volume

def get_flipped_triangles(vertices, faces):
    """ Given set of 2D vertices, find triangles that have flipped orientation -- i.e. negative determinant """
    # For each original 3D triangle, compute the normal
    from meshing.mesh import Mesh
    from meshing.analysis import computeFaceNormals

    # Old and new vertices must be of same dimension -- otherwise comparison makes no sense!
    assert vertices.shape[1] == 2, f"Vertices must be 2D! {vertices.shape[1]} != 2"

    # Compute cross product of edge vectors => signed area in 2D
    mesh = Mesh(vertices, faces)
    computeFaceNormals(mesh)

    # Flipped triangles have negative determinant
    flipped = np.where(mesh.fnormals < 0)[0]

    return flipped

class DifferentiableThreshold(torch.autograd.Function):
    """
    In the forward pass this operation behaves like a threshold (> epsilon => 1, else 0).
    But in the backward pass its gradient is 1 everywhere the threshold is passed.
    """

    @staticmethod
    # @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    # @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)