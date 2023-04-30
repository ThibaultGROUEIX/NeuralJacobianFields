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
def tutte_embedding(vertices, faces, fixclosed=False):
    import igl
    bnd = igl.boundary_loop(faces)

    ## If mesh is closed then we cut a seam if set
    if fixclosed and (bnd is None or len(bnd) == 0):
        from meshing.mesh import Mesh
        from meshing.edit import EdgeCut

        # TODO: cut out a triangle
        mesh = Mesh(vertices, faces)
        # for he in mesh.topology.faces[0].adjacentHalfedges():
        #     EdgeCut(mesh, he.index).apply()

        he2 = mesh.topology.halfedges[0].next.index
        EdgeCut(mesh, 0).apply()
        EdgeCut(mesh, he2).apply()
        bnd = igl.boundary_loop(faces)
    elif bnd is None:
        raise ValueError(f"tutte_embedding: mesh has no boundary and fixclosed is not set!")

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    ## Harmonic parametrization for the internal vertices
    assert not np.isnan(bnd).any(), f"NaN found in boundary loop!"
    assert not np.isnan(bnd_uv).any(), f"NaN found in tutte initialized UVs!"
    uv_init = igl.harmonic_weights(vertices, faces, bnd, np.array(bnd_uv, dtype=vertices.dtype), 1)

    return uv_init

# Convert each triangle into local coordinates: A -> (0,0), B -> (x2, 0), C -> (x3, y3)
def get_local_tris(vertices, faces, device=torch.device("cpu")):
    fverts = vertices[faces].to(device)
    e1 = fverts[:, 1, :] - fverts[:, 0, :]
    e2 = fverts[:, 2, :] - fverts[:, 0, :]
    s = torch.linalg.norm(e1, dim=1)
    t = torch.linalg.norm(e2, dim=1)
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1))
    x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)])
    y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)])
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

    # Keep track of all split correspondences
    # splitlog = defaultdict(list)
    # # NOTE: this is same order as export_edge_face_connectivity()
    # boundarycount = 0
    # for ei, edge in sorted(ogmesh.topology.edges.items()):
    #     if edge.onBoundary():
    #         boundarycount += 1
    #         continue
    #     if edgedist[ei - boundarycount] > epsilon: # edgedist removes boundaries
    #         # New face indices
    #         replacei = {edge.halfedge.vertex.index: len(ogvs) + len(newvs), edge.halfedge.tip_vertex().index: len(ogvs) + len(newvs) + 1}
    #         newfi = []
    #         for v in edge.halfedge.twin.face.adjacentVertices():
    #             if v.index in replacei.keys():
    #                 newfi.append(replacei[v.index])
    #             else:
    #                 newfi.append(v.index)
    #         newfi = torch.tensor(newfi).long().to(fs.device)
    #         newfs[edge.halfedge.twin.face.index] = newfi

    #         # Record in split log
    #         splitlog[edge.halfedge.vertex.index].append(len(ogvs) + len(newvs))
    #         splitlog[edge.halfedge.tip_vertex().index].append(len(ogvs) + len(newvs) + 1)

    #         # New vertices (just copy over adjacent)
    #         newvs.extend([vs[edge.halfedge.vertex.index], vs[edge.halfedge.tip_vertex().index]])

    #         if return_cut_edges:
    #             cut_es.append(ei)

    #         if return_cut_length:
    #             cutlen += ogmesh.length(edge)
    #     else:
    #         # NOTE: EDGE CASE -- if one of the vertex edges is already new, then need to copy the NEW INDEX OVER
    #         # Check if one of the adjacent faces has a split vertex
    #         v1log = splitlog[edge.halfedge.vertex.index]
    #         v2log = splitlog[edge.halfedge.tip_vertex().index]
    #         f1 = newfs[edge.halfedge.face.index]
    #         f2 = newfs[edge.halfedge.twin.face.index]

    #         replacei = {}
    #         replacef1_count = 0
    #         for v in f1:
    #             if v in v1log:
    #                 replacei[edge.halfedge.vertex.index] = v.item()
    #                 replacef1_count += 1
    #             elif v in v2log:
    #                 replacei[edge.halfedge.tip_vertex().index] = v.item()
    #                 replacef1_count += 1

    #         replacef2_count = 0
    #         for v in f2:
    #             if v in v1log:
    #                 replacei[edge.halfedge.vertex.index] = v.item()
    #                 replacef2_count += 1
    #             elif v in v2log:
    #                 replacei[edge.halfedge.tip_vertex().index] = v.item()
    #                 replacef2_count += 1

    #         if len(replacei) > 0:
    #             # NOTE: Its possible for split vertices to be merged back again
    #             # Edge case: if there are two replacement vertices, then it is one from each face
    #             if len(replacei) == 2:
    #                 assert replacef1_count == 1 and replacef2_count == 1, f"When two split vertices, then must be one from each face."
    #             else:
    #                 assert len(replacei) == 1, f"Found more than one split vertex! {replacei}"

    #             newf1 = []
    #             newf2 = []
    #             for v in f1:
    #                 if v.item() in replacei.keys():
    #                     newf1.append(replacei[v.item()])
    #                 else:
    #                     newf1.append(v.item())
    #             for v in f2:
    #                 if v.item() in replacei.keys():
    #                     newf2.append(replacei[v.item()])
    #                 else:
    #                     newf2.append(v.item())
    #             newf1 = torch.tensor(newf1).long().to(fs.device)
    #             newfs[edge.halfedge.face.index] = newf1
    #             newf2 = torch.tensor(newf2).long().to(fs.device)
    #             newfs[edge.halfedge.twin.face.index] = newf2

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
def leastSquaresTranslation(vertices, faces, trisoup, iterate=False, debug=False, patience=5):
    """ vertices: V x 3 np array
        faces: F x 3 np array
        trisoup: F x 3 x 2 np array

        returns: F x 2 numpy array with optimized translations per triangle"""
    from meshing.mesh import Mesh

    # Build A, B matrices using the edge connectivity
    mesh = Mesh(vertices, faces)
    fconn, vconn = mesh.topology.export_edge_face_connectivity(faces)
    fconn = np.array(fconn, dtype=int) # E x {f0, f1}
    vconn = np.array(vconn, dtype=int) # E x [[v0_1,v1_1], [v0_2, v1_2]]

    A = np.zeros((len(fconn) * 2, len(faces)))

    # Duplicate every row of fconn (bc each edge involves two vertex pairs)
    edge_finds = np.repeat(fconn, 2, axis=0)
    A[np.arange(len(A)),edge_finds[:,0]] = 1 # distance vectors go f0 -> f1
    A[np.arange(len(A)),edge_finds[:,1]] = -1

    ef0 = trisoup[fconn[:,[0]], vconn[:,0]] # E x 2 x 2
    ef1 = trisoup[fconn[:,[1]], vconn[:,1]] # E x 2 x 2

    B = (ef1 - ef0).reshape(2 * len(fconn), 2)

    if debug:
        import polyscope as ps
        ps.init()
        ps_soup = ps.register_surface_mesh("initial soup", trisoup.reshape(-1, 2), np.arange(len(faces) * 3).reshape(-1, 3), edge_width=1)

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
            opttrans, residuals, rank, s = np.linalg.lstsq(weights * A, weights * B, rcond=None)
            finaltrans += opttrans.reshape(-1, 1, 2)
            newsoup = trisoup + finaltrans

            # If debugging, then visualize
            # if debug:
            #     ps_soup = ps.register_surface_mesh(f"soup {tot_count}", newsoup.reshape(-1, 2), np.arange(len(faces) * 3).reshape(-1, 3), edge_width=1,
            #                                        enabled=True)
            #     ps.show()

            # Recompute edge distances and associated weights
            ef0 = newsoup[fconn[:,[0]], vconn[:,0]] # E x 2 x 2
            ef1 = newsoup[fconn[:,[1]], vconn[:,1]] # E x 2 x 2
            B = (ef1 - ef0).reshape(2 * len(fconn), 2) # E * 2 x 2

            # Weights are 1/(correspondence distances)
            # TODO: maybe we should sum over faces instead
            weights = 1/(np.linalg.norm(B, axis=1, keepdims=1) + 1e-10)
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
            print(f"L0 least squares translation: done with step {tot_count}. Stationary count: {stationary_count}.")
    else:
        opttrans, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        finaltrans = opttrans.reshape(-1, 1, 2)

    if debug:
        ps_soup = ps.register_surface_mesh("final soup", newsoup.reshape(-1, 2), np.arange(len(faces) * 3).reshape(-1, 3), edge_width=1, enabled=True)
        ps.show()

    return finaltrans

# ====================== UV Stuff ========================
def make_cut(mesh, cutlist, b0=None, b1=None):
    from meshing.edit import EdgeCut

    for i in range(len(cutlist)-1):
        # Cut: 1 new vertex, 1 new edge, two new halfedges
        # Find the halfedge associated with each successive cut
        vsource = cutlist[i]
        vtarget = cutlist[i+1]

        # Instead of assert, just continue if broken
        if not mesh.topology.vertices[vsource].onBoundary():
            continue
        if not vtarget in [v.index for v in mesh.topology.vertices[vsource].adjacentVertices()]:
            continue
        # assert mesh.topology.vertices[vsource].onBoundary()
        # assert vtarget in [v.index for v in mesh.topology.vertices[vsource].adjacentVertices()]

        for he in mesh.topology.vertices[vsource].adjacentHalfedges():
            if he.tip_vertex().index == vtarget:
                break
        edt = EdgeCut(mesh, he.index)
        edt.apply()
        del edt

        # assert np.all(mesh.vertices[vsource] == mesh.vertices[-1])

    # b0 and b1 should iterate through same set of vertices
    if b0 is not None and b1 is not None:
        b0_v = set([v.index for v in mesh.topology.boundaries[b0].adjacentVertices()])
        b1_v = set([v.index for v in mesh.topology.boundaries[b1].adjacentVertices()])

        # Debugging
        if b0_v != b1_v:
            import polyscope as ps
            ps.init()
            ps.remove_all_structures()
            ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces, edge_width=1)
            b0_colors = np.zeros(len(mesh.vertices))
            b0_colors[list(b0_v)] = 1
            b1_colors = np.zeros(len(mesh.vertices))
            b1_colors[list(b1_v)] = 1
            ps_mesh.add_scalar_quantity("b0", b0_colors, enabled=True)
            ps_mesh.add_scalar_quantity("b1", b1_colors, enabled=True)
            ps.show()

        # assert b0_v == b1_v, f"Boundaries {b0} and {b1} do not coincide after cut!"

        # Delete second boundary
        del mesh.topology.boundaries[b1]

def cut_to_disk(mesh, verbose=False):
    count = 0

    # Don't allow cut if mesh has isolated faces
    if mesh.topology.hasIsolatedFaces():
        return

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

        # Compute shortest paths from current boundary vertices to all other boundary vertices
        b_vs = np.array([v.index for b in avail_b for v in mesh.topology.boundaries[b].adjacentVertices()])
        # Sometimes two boundaries share vertex
        # b_vs = np.array(list(set(b_vs).difference(set(subboundary_vs))))

        if len(b_vs) == 0:
            print(f"Overlapping boundaries!")
            break

        if verbose:
            print(f"Iteration {count}: graph construct time {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        # Heuristic: initialize to first vertex in subboundary, and compute all shortest paths to all other boundaries
        # Choose shortest path and cut
        cutlists = []
        for init_v in subboundary_vs:
            tmpcutlists = graph.get_shortest_paths(init_v, b_vs, edgeweights)
            if len(tmpcutlists) > 0:
                cutlists.extend(tmpcutlists)
        if len(cutlists) == 0:
            print("No more paths found.")
            break
        # Remove all 0 length paths
        cutlists = [cutlist for cutlist in cutlists if len(cutlist) > 0]
        if len(cutlists) == 0:
            print("No more paths found.")
            break
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
        make_cut(mesh, cutlist, current_b, next_b)
        count += 1
        if verbose:
            print(f"Iteration {count}: cutting took {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        # if mesh.topology.hasNonManifoldVertices() or mesh.topology.hasNonManifoldEdges():
        #     print(f"Mesh became non-manifold from cuts!")
        #     break

        if mesh.topology.hasNonManifoldEdges():
            print(f"Mesh became non-manifold from cuts!")
            break

        if count >= 10:
            print(f"Cuts infinite loop.")
            break

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

def tutte_embedding(vertices, faces):
    import igl
    bnd = igl.boundary_loop(faces)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    ## Harmonic parametrization for the internal vertices
    assert not np.isnan(bnd).any()
    assert not np.isnan(bnd_uv).any()
    uv_init = igl.harmonic_weights(vertices, faces, bnd, np.array(bnd_uv, dtype=vertices.dtype), 1)

    return uv_init

def SLIM(mesh, v_with_holes = None, f_with_holes = None):
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
    slim.solve(500)
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

def run_slim(mesh, cut=True, verbose=False, time=False):
    did_cut = False
    if mesh.topology.hasNonManifoldEdges():
        print(f"run_slim: Non-manifold edges found.")
        return None, None, did_cut

    if cut:
        if time:
            import time
            t0 = time.time()

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

        if len(mesh.topology.boundaries) > 1:
            cut_to_disk(mesh, verbose)
            did_cut = True

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

        if not hasattr(mesh, "vertexangle"):
            from meshing.analysis import computeVertexAngle
            computeVertexAngle(mesh)

        # Cut cones
        # Only cut cones if one boundary exists
        singlevs = np.where(2 * np.pi - mesh.vertexangle >= np.pi/2)[0]
        if len(singlevs) >= 0 and len(mesh.topology.boundaries) == 1: # Edge case: no boundaries
            cut_to_disk_single(mesh, singlevs, verbose)
            did_cut = True

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

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