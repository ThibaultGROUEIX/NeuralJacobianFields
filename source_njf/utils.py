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